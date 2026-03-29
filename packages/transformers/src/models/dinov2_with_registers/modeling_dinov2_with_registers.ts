import { SequenceClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class Dinov2WithRegistersPreTrainedModel extends PreTrainedModel {}

/**
 * The bare Dinov2WithRegisters Model transformer outputting raw hidden-states without any specific head on top.
 */
export class Dinov2WithRegistersModel extends Dinov2WithRegistersPreTrainedModel {}

/**
 * Dinov2WithRegisters Model transformer with an image classification head on top (a linear layer on top of the final hidden state of the [CLS] token) e.g. for ImageNet.
 */
export class Dinov2WithRegistersForImageClassification extends Dinov2WithRegistersPreTrainedModel {
    async _call(model_inputs: Record<string, any>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
