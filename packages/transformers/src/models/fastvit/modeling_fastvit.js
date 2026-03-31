import { SequenceClassifierOutput } from '../modeling_outputs.js';
import { PreTrainedModel } from '../modeling_utils.js';

export class FastViTPreTrainedModel extends PreTrainedModel {}
export class FastViTModel extends FastViTPreTrainedModel {}
export class FastViTForImageClassification extends FastViTPreTrainedModel {
    /**
     * @param {Record<string, unknown>} model_inputs
     */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
