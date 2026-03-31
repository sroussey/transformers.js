import { SequenceClassifierOutput } from '../modeling_outputs.js';
import { PreTrainedModel } from '../modeling_utils.js';

export class PvtPreTrainedModel extends PreTrainedModel {}
export class PvtModel extends PvtPreTrainedModel {}
export class PvtForImageClassification extends PvtPreTrainedModel {
    /** @param {Object} model_inputs */
    async _call(model_inputs) {
        return new SequenceClassifierOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}
