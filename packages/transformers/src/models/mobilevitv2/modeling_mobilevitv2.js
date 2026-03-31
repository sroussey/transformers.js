import { SequenceClassifierOutput } from '../modeling_outputs.js';
import { PreTrainedModel } from '../modeling_utils.js';

export class MobileViTV2PreTrainedModel extends PreTrainedModel {}
export class MobileViTV2Model extends MobileViTV2PreTrainedModel {}
export class MobileViTV2ForImageClassification extends MobileViTV2PreTrainedModel {
    /**
     * @param {Object} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
// TODO: MobileViTV2ForSemanticSegmentation
