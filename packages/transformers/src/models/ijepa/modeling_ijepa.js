import { SequenceClassifierOutput } from '../modeling_outputs.js';
import { PreTrainedModel } from '../modeling_utils.js';

export class IJepaPreTrainedModel extends PreTrainedModel {}
export class IJepaModel extends IJepaPreTrainedModel {}
export class IJepaForImageClassification extends IJepaPreTrainedModel {
    /**
     * @param {Object} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
