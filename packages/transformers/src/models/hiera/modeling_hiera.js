import { SequenceClassifierOutput } from '../modeling_outputs.js';
import { PreTrainedModel } from '../modeling_utils.js';

export class HieraPreTrainedModel extends PreTrainedModel {}
export class HieraModel extends HieraPreTrainedModel {}
export class HieraForImageClassification extends HieraPreTrainedModel {
    /** @param {Object} model_inputs */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
