import { SequenceClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class BeitPreTrainedModel extends PreTrainedModel {}
export class BeitModel extends BeitPreTrainedModel {}
export class BeitForImageClassification extends BeitPreTrainedModel {
    /**
     * @param {Object} model_inputs
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
