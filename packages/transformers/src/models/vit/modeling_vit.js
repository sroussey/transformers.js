import { SequenceClassifierOutput } from '../modeling_outputs.js';
import { PreTrainedModel } from '../modeling_utils.js';

export class ViTPreTrainedModel extends PreTrainedModel {}
export class ViTModel extends ViTPreTrainedModel {}
export class ViTForImageClassification extends ViTPreTrainedModel {
    /**
     * @param {Record<string, unknown>} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
