import { SequenceClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class MobileViTPreTrainedModel extends PreTrainedModel {}
export class MobileViTModel extends MobileViTPreTrainedModel {}
export class MobileViTForImageClassification extends MobileViTPreTrainedModel {
    /**
     * @param {Record<string, unknown>} model_inputs
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
// TODO: MobileViTForSemanticSegmentation
