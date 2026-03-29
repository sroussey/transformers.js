import { SequenceClassifierOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class ConvNextPreTrainedModel extends PreTrainedModel {}

/**
 * The bare ConvNext model outputting raw features without any specific head on top.
 */
export class ConvNextModel extends ConvNextPreTrainedModel {}

/**
 * ConvNext Model with an image classification head on top (a linear layer on top of the pooled features), e.g. for ImageNet.
 */
export class ConvNextForImageClassification extends ConvNextPreTrainedModel {
    /**
     * @param {Object} model_inputs
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new SequenceClassifierOutput(await super._call(model_inputs));
    }
}
