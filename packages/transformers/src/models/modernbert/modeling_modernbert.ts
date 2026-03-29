import { MaskedLMOutput, SequenceClassifierOutput, TokenClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class ModernBertPreTrainedModel extends PreTrainedModel {}
export class ModernBertModel extends ModernBertPreTrainedModel {}

export class ModernBertForMaskedLM extends ModernBertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<MaskedLMOutput>} An object containing the model's output logits for masked language modeling.
     */
    async _call(model_inputs: Record<string, any>) {
        return new MaskedLMOutput(await super._call(model_inputs));
    }
}

export class ModernBertForSequenceClassification extends ModernBertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} An object containing the model's output logits for sequence classification.
     */
    async _call(model_inputs: Record<string, any>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}

export class ModernBertForTokenClassification extends ModernBertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<TokenClassifierOutput>} An object containing the model's output logits for token classification.
     */
    async _call(model_inputs: Record<string, any>) {
        return new TokenClassifierOutput(await super._call(model_inputs));
    }
}
