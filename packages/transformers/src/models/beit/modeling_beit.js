import { SequenceClassifierOutput } from '../modeling_outputs.js';
import { PreTrainedModel } from '../modeling_utils.js';

export class BeitPreTrainedModel extends PreTrainedModel {}
export class BeitModel extends BeitPreTrainedModel {}
export class BeitForImageClassification extends BeitPreTrainedModel {
    /**
     * @param {Object} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
