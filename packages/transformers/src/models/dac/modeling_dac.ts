import { Tensor } from '../../utils/tensor';
import { ModelOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';
import { sessionRun } from '../session';

export class DacEncoderOutput extends ModelOutput {
    audio_codes;
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.audio_codes Discrete code embeddings, of shape `(batch_size, num_quantizers, codes_length)`.
     */
    constructor({ audio_codes }: { audio_codes: Tensor }) {
        super();
        this.audio_codes = audio_codes;
    }
}

export class DacDecoderOutput extends ModelOutput {
    audio_values;
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.audio_values Decoded audio values, of shape `(batch_size, num_channels, sequence_length)`.
     */
    constructor({ audio_values }: { audio_values: Tensor }) {
        super();
        this.audio_values = audio_values;
    }
}

export class DacPreTrainedModel extends PreTrainedModel {
    main_input_name = 'input_values';
    forward_params = ['input_values'];
}

/**
 * The DAC (Descript Audio Codec) model.
 */
export class DacModel extends DacPreTrainedModel {
    /**
     * Encodes the input audio waveform into discrete codes.
     * @param {Object} inputs Model inputs
     * @param {Tensor} [inputs.input_values] Float values of the input audio waveform, of shape `(batch_size, channels, sequence_length)`).
     * @returns {Promise<DacEncoderOutput>} The output tensor of shape `(batch_size, num_codebooks, sequence_length)`.
     */
    async encode(inputs: Record<string, Tensor>) {
        return new DacEncoderOutput(await sessionRun(this.sessions['encoder_model'], inputs) as { audio_codes: Tensor });
    }

    /**
     * Decodes the given frames into an output audio waveform.
     * @param {DacEncoderOutput} inputs The encoded audio codes.
     * @returns {Promise<DacDecoderOutput>} The output tensor of shape `(batch_size, num_channels, sequence_length)`.
     */
    async decode(inputs: DacEncoderOutput) {
        return new DacDecoderOutput(await sessionRun(this.sessions['decoder_model'], inputs as unknown as Record<string, Tensor>) as { audio_values: Tensor });
    }
}

export class DacEncoderModel extends DacPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path: string, options: Record<string, unknown> = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: (options.model_file_name as string) ?? 'encoder_model',
        });
    }
}
export class DacDecoderModel extends DacPreTrainedModel {
    /** @type {typeof PreTrainedModel.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path: string, options: Record<string, unknown> = {}) {
        return super.from_pretrained(pretrained_model_name_or_path, {
            ...options,
            // Update default model file name if not provided
            model_file_name: (options.model_file_name as string) ?? 'decoder_model',
        });
    }
}
