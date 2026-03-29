import { SequenceClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class HieraPreTrainedModel extends PreTrainedModel {}
export class HieraModel extends HieraPreTrainedModel {}
export class HieraForImageClassification extends HieraPreTrainedModel {
    async _call(model_inputs: Record<string, any>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
