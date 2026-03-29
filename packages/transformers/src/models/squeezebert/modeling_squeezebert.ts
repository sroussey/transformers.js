import { MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class SqueezeBertPreTrainedModel extends PreTrainedModel {}
export class SqueezeBertModel extends SqueezeBertPreTrainedModel {}
export class SqueezeBertForMaskedLM extends SqueezeBertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<MaskedLMOutput>} returned object
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new MaskedLMOutput(await super._call(model_inputs));
    }
}
export class SqueezeBertForSequenceClassification extends SqueezeBertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} returned object
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
export class SqueezeBertForQuestionAnswering extends SqueezeBertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<QuestionAnsweringModelOutput>} returned object
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new QuestionAnsweringModelOutput(await super._call(model_inputs));
    }
}
