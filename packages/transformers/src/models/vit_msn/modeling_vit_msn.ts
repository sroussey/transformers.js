import { SequenceClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class ViTMSNPreTrainedModel extends PreTrainedModel {}
export class ViTMSNModel extends ViTMSNPreTrainedModel {}
export class ViTMSNForImageClassification extends ViTMSNPreTrainedModel {
    /**
     * @param {Object} model_inputs
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
