import { PreTrainedModel, default_merge_input_ids_with_image_features } from '../modeling_utils';

export class LlavaPreTrainedModel extends PreTrainedModel {
    forward_params = ['input_ids', 'attention_mask', 'pixel_values', 'position_ids', 'past_key_values'];
}

/**
 * The LLAVA model which consists of a vision backbone and a language model.
 */
export class LlavaForConditionalGeneration extends LlavaPreTrainedModel {
    _merge_input_ids_with_image_features(kwargs: Record<string, any>) {
        const vision_hidden_size = kwargs.image_features.dims.at(-1);
        const reshaped_image_hidden_states = kwargs.image_features.view(-1, vision_hidden_size);

        return default_merge_input_ids_with_image_features({
            image_token_id: (this.config.image_token_index ?? this.config.image_token_id) as number,
            ...kwargs,
            image_features: reshaped_image_hidden_states,
        } as Parameters<typeof default_merge_input_ids_with_image_features>[0]);
    }
}

export class Moondream1ForConditionalGeneration extends LlavaForConditionalGeneration {} // NOTE: extends LlavaForConditionalGeneration

export class LlavaQwen2ForCausalLM extends LlavaForConditionalGeneration {} // NOTE: extends LlavaForConditionalGeneration
