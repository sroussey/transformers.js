import { MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput } from '../modeling_outputs.js';
import { PreTrainedModel } from '../modeling_utils.js';

export class SqueezeBertPreTrainedModel extends PreTrainedModel {}
export class SqueezeBertModel extends SqueezeBertPreTrainedModel {}
export class SqueezeBertForMaskedLM extends SqueezeBertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<MaskedLMOutput>} returned object
     */
    async _call(model_inputs) {
        return new MaskedLMOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
export class SqueezeBertForSequenceClassification extends SqueezeBertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<SequenceClassifierOutput>} returned object
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
export class SqueezeBertForQuestionAnswering extends SqueezeBertPreTrainedModel {
    /**
     * Calls the model on new inputs.
     *
     * @param {Object} model_inputs The inputs to the model.
     * @returns {Promise<QuestionAnsweringModelOutput>} returned object
     */
    async _call(model_inputs) {
        return new QuestionAnsweringModelOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
