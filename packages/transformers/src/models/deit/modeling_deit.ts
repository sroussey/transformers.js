import { SequenceClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class DeiTPreTrainedModel extends PreTrainedModel {}
export class DeiTModel extends DeiTPreTrainedModel {}
export class DeiTForImageClassification extends DeiTPreTrainedModel {
    async _call(model_inputs: Record<string, any>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
