import { SequenceClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class PvtPreTrainedModel extends PreTrainedModel {}
export class PvtModel extends PvtPreTrainedModel {}
export class PvtForImageClassification extends PvtPreTrainedModel {
    async _call(model_inputs: Record<string, any>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
