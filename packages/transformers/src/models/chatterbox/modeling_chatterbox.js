import { cat, full, ones, Tensor } from '../../utils/tensor.js';
import { decoder_forward, decoder_prepare_inputs_for_generation, PreTrainedModel } from '../modeling_utils.js';
import { sessionRun } from '../session.js';

/**
 * @typedef {import('../../generation/configuration_utils.js').GenerationConfig} GenerationConfig
 * @typedef {import('../../generation/parameters.js').GenerationFunctionParameters} GenerationFunctionParameters
 */

const SILENCE_TOKEN = 4299n;
const START_SPEECH_TOKEN = 6561n;

export class ChatterboxPreTrainedModel extends PreTrainedModel {
    forward_params = [
        'input_ids',
        'inputs_embeds',
        'attention_mask',
        'position_ids',
        'audio_values',
        'exaggeration',
        'audio_features',
        'audio_tokens',
        'speaker_embeddings',
        'speaker_features',
        'past_key_values',
    ];
    main_input_name = 'input_ids';

    _return_dict_in_generate_keys = ['audio_tokens', 'speaker_embeddings', 'speaker_features'];
}
export class ChatterboxModel extends ChatterboxPreTrainedModel {
    /**
     * @param {Tensor} audio_values
     * @returns {Promise<{audio_features: Tensor, audio_tokens: Tensor, speaker_embeddings: Tensor, speaker_features: Tensor}>}
     */
    async encode_speech(audio_values) {
        return /** @type {any} */ (sessionRun(this.sessions['speech_encoder'], {
            audio_values,
        }));
    }

    async forward({
        // Produced by the tokenizer/processor:
        input_ids = /** @type {Tensor | null} */ (null),
        attention_mask = /** @type {Tensor | null} */ (null),
        audio_values = /** @type {Tensor | null} */ (null),
        exaggeration = /** @type {Tensor | number | number[] | null} */ (null),

        // Used during generation:
        position_ids = /** @type {Tensor | null} */ (null),
        inputs_embeds = /** @type {Tensor | null} */ (null),
        past_key_values = /** @type {import('../../cache_utils.js').DynamicCache | null} */ (null),

        // Generic generation parameters
        generation_config = /** @type {import('../../generation/configuration_utils.js').GenerationConfig | null} */ (null),
        logits_processor = /** @type {import('../../generation/logits_process.js').LogitsProcessorList | null} */ (null),

        // Speaker embeddings/features (useful for re-using pre-computed speaker data)
        audio_features = /** @type {Tensor | null} */ (null), // float32[batch_size,sequence_length,1024]
        audio_tokens = /** @type {Tensor | null} */ (null), // int64[batch_size,audio_sequence_length]
        speaker_embeddings = /** @type {Tensor | null} */ (null), // float32[batch_size,192]
        speaker_features = /** @type {Tensor | null} */ (null), // float32[batch_size,feature_dim,80]

        // TODO: needed?
        ...kwargs
    }) {
        let speech_encoder_outputs;
        if (!inputs_embeds) {
            const expected_inputs = this.sessions['embed_tokens'].inputNames;
            /** @type {Record<string, Tensor>} */
            const embed_model_inputs = { input_ids: /** @type {Tensor} */ (input_ids) };
            if (expected_inputs.includes('exaggeration')) {
                // Support the following types for exaggeration:
                // 1. null/undefined (no exaggeration): use the default of 0.5
                // 2. number: broadcast to (batch_size,)
                // 3. number[]: convert to Tensor of shape (batch_size,)
                // 4. Tensor of shape (batch_size, 1)
                if (!(exaggeration instanceof Tensor)) {
                    const batch_size = /** @type {Tensor} */ (input_ids).dims[0];
                    if (exaggeration == null) {
                        exaggeration = full([batch_size], 0.5);
                    } else if (typeof exaggeration === 'number') {
                        exaggeration = full([batch_size], exaggeration);
                    } else if (Array.isArray(exaggeration)) {
                        exaggeration = new Tensor('float32', exaggeration, [batch_size]);
                    } else {
                        throw new Error('Unsupported type for `exaggeration` input');
                    }
                }
                embed_model_inputs.exaggeration = exaggeration;
            }
            if (expected_inputs.includes('position_ids')) {
                embed_model_inputs.position_ids = /** @type {Tensor} */ (position_ids);
            }

            ({ inputs_embeds } = /** @type {any} */ (await sessionRun(this.sessions['embed_tokens'], embed_model_inputs)));

            if (audio_features && audio_tokens && speaker_embeddings && speaker_features) {
                // Use pre-computed speech encoder outputs
                speech_encoder_outputs = { audio_features, audio_tokens, speaker_embeddings, speaker_features };
            }

            if (speech_encoder_outputs || audio_values) {
                speech_encoder_outputs ??= await this.encode_speech(/** @type {Tensor} */ (audio_values));

                // Update LLM inputs
                inputs_embeds = cat([speech_encoder_outputs.audio_features, /** @type {Tensor} */ (inputs_embeds)], 1);
                attention_mask = ones([inputs_embeds.dims[0], inputs_embeds.dims[1]]);
            } else {
                const target_length = /** @type {Tensor} */ (inputs_embeds).dims[1];
                if (!past_key_values || target_length !== 1) {
                    throw new Error('Incorrect state encountered during generation.');
                }
                const past_length = past_key_values.get_seq_length();
                attention_mask = ones([/** @type {Tensor} */ (inputs_embeds).dims[0], past_length + target_length]);
            }
        }

        const outputs = await decoder_forward(
            this,
            {
                inputs_embeds,
                past_key_values,
                attention_mask,
                generation_config,
                logits_processor,
            },
            false,
        );
        return {
            ...outputs,
            ...speech_encoder_outputs,
        };
    }

    /**
     * @param {bigint[][]} input_ids
     * @param {Record<string, any>} model_inputs
     * @param {Record<string, unknown>} generation_config
     */
    prepare_inputs_for_generation(input_ids, model_inputs, generation_config) {
        if (!model_inputs.position_ids && this.sessions['embed_tokens'].inputNames.includes('position_ids')) {
            // If position_ids are not provided, we create them on the fly using the position of the START_SPEECH_TOKEN
            if (/** @type {Tensor} */ (model_inputs.input_ids).dims[1] === 1) {
                const position_ids = Array.from(
                    {
                        length: input_ids.length,
                    },
                    (/** @type {unknown} */ _, /** @type {number} */ i) => input_ids[i].length - input_ids[i].findLastIndex((/** @type {bigint} */ x) => x == START_SPEECH_TOKEN) - 1,
                );
                model_inputs.position_ids = new Tensor('int64', position_ids, [input_ids.length, 1]);
            } else {
                const batched_input_ids = /** @type {bigint[][]} */ (/** @type {Tensor} */ (model_inputs.input_ids).tolist());
                const position_ids_list = batched_input_ids.map((/** @type {bigint[]} */ ids) => {
                    let position = 0;
                    return ids.map((/** @type {bigint} */ id) => (id >= START_SPEECH_TOKEN ? 0 : position++));
                });
                model_inputs.position_ids = new Tensor('int64', position_ids_list.flat(), /** @type {Tensor} */ (model_inputs.input_ids).dims);
            }
        }
        if (/** @type {Tensor} */ (model_inputs.input_ids).dims[1] === 1) {
            // We are in generation mode and no longer need the audio inputs
            delete model_inputs.audio_values;
            delete model_inputs.audio_features;
            delete model_inputs.audio_tokens;
            delete model_inputs.speaker_embeddings;
            delete model_inputs.speaker_features;
        }
        return decoder_prepare_inputs_for_generation(this, input_ids, model_inputs, /** @type {GenerationConfig} */ (/** @type {unknown} */ (generation_config)));
    }

    /** @type {PreTrainedModel['generate']} */
    async generate(/** @type {GenerationFunctionParameters} */ params) {
        const _result = /** @type {Record<string, Tensor>} */ (/** @type {unknown} */ (await super.generate({
            ...params,
            return_dict_in_generate: true,
        })));
        const { sequences, audio_tokens, speaker_embeddings, speaker_features } = _result;

        const new_tokens = sequences.slice(null, [
            params.input_ids.dims[1], // Exclude start of speech token
            -1, // Exclude end of speech token
        ]);

        const silence_tokens = full([new_tokens.dims[0], 3], SILENCE_TOKEN); // Add 3 silence tokens
        const speech_tokens = cat([audio_tokens, new_tokens, silence_tokens], 1);

        const { waveform } = /** @type {any} */ (await sessionRun(this.sessions['conditional_decoder'], {
            speech_tokens,
            speaker_features,
            speaker_embeddings,
        }));
        return waveform;
    }
}
