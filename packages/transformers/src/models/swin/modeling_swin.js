import { SequenceClassifierOutput } from '../modeling_outputs.js';
import { PreTrainedModel } from '../modeling_utils.js';

export class SwinPreTrainedModel extends PreTrainedModel {}
export class SwinModel extends SwinPreTrainedModel {}
export class SwinForImageClassification extends SwinPreTrainedModel {
    /** @param {Object} model_inputs */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
export class SwinForSemanticSegmentation extends SwinPreTrainedModel {}
