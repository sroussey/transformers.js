import { SequenceClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class FastViTPreTrainedModel extends PreTrainedModel {}
export class FastViTModel extends FastViTPreTrainedModel {}
export class FastViTForImageClassification extends FastViTPreTrainedModel {
    /**
     * @param {Record<string, unknown>} model_inputs
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
