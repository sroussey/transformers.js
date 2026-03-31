import { Tensor } from '../../utils/tensor.js';
import { ModelOutput } from '../modeling_outputs.js';
import { PreTrainedModel } from '../modeling_utils.js';
import { sessionRun } from '../session.js';

export class MimiEncoderOutput extends ModelOutput {
    audio_codes;
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.audio_codes Discrete code embeddings, of shape `(batch_size, num_quantizers, codes_length)`.
     */
    constructor({ audio_codes }) {
        super();
        this.audio_codes = audio_codes;
    }
}

export class MimiDecoderOutput extends ModelOutput {
    audio_values;
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.audio_values Decoded audio values, of shape `(batch_size, num_channels, sequence_length)`.
     */
    constructor({ audio_values }) {
        super();
        this.audio_values = audio_values;
    }
}

export class MimiPreTrainedModel extends PreTrainedModel {
    main_input_name = 'input_values';
    forward_params = ['input_values'];
}

/**
 * The Mimi neural audio codec model.
 */
export class MimiModel extends MimiPreTrainedModel {
    /**
     * Encodes the input audio waveform into discrete codes.
     * @param {Object} inputs Model inputs
     * @param {Tensor} [inputs.input_values] Float values of the input audio waveform, of shape `(batch_size, channels, sequence_length)`).
     * @returns {Promise<MimiEncoderOutput>} The output tensor of shape `(batch_size, num_codebooks, sequence_length)`.
     */
    async encode(inputs) {
        return new MimiEncoderOutput(/** @type {{ audio_codes: Tensor }} */ (await sessionRun(this.sessions['encoder_model'], inputs)));
    }

    /**
     * Decodes the given frames into an output audio waveform.
     * @param {MimiEncoderOutput} inputs The encoded audio codes.
     * @returns {Promise<MimiDecoderOutput>} The output tensor of shape `(batch_size, num_channels, sequence_length)`.
     */
    async decode(inputs) {
        return new MimiDecoderOutput(/** @type {{ audio_values: Tensor }} */ (await sessionRun(this.sessions['decoder_model'], /** @type {Record<string, Tensor>} */ (/** @type {unknown} */ (inputs)))));
    }
}

export class MimiEncoderModel extends MimiPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: /** @type {string} */ (options.model_file_name) ?? 'encoder_model',
        });
    }
}
export class MimiDecoderModel extends MimiPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: /** @type {string} */ (options.model_file_name) ?? 'decoder_model',
        });
    }
}
