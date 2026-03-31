import { PreTrainedModel, default_merge_input_ids_with_audio_features } from '../modeling_utils.js';

export class UltravoxPreTrainedModel extends PreTrainedModel {
    forward_params = ['input_ids', 'attention_mask', 'position_ids', 'audio_values', 'past_key_values'];
}

export class UltravoxModel extends UltravoxPreTrainedModel {
    /** @param {Record<string, unknown> & { audio_features: import('../../utils/tensor.js').Tensor }} kwargs */
    _merge_input_ids_with_audio_features(kwargs) {
        const audio_hidden_size = kwargs.audio_features.dims.at(-1);
        const reshaped_audio_features = kwargs.audio_features.view(-1, audio_hidden_size);

        return default_merge_input_ids_with_audio_features(/** @type {any} */ ({
            audio_token_id: /** @type {number} */ (/** @type {any} */ (this.config).ignore_index ?? /** @type {any} */ (this.config).audio_token_id ?? /** @type {any} */ (this.config).audio_token_index),
            ...kwargs,
            audio_features: reshaped_audio_features,
        }));
    }
}
