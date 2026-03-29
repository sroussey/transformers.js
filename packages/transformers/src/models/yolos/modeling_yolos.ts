import { Tensor } from '../../utils/tensor';
import { ModelOutput } from '../modeling_outputs';
import { PreTrainedModel } from '../modeling_utils';

export class YolosPreTrainedModel extends PreTrainedModel {}
export class YolosModel extends YolosPreTrainedModel {}
export class YolosForObjectDetection extends YolosPreTrainedModel {
    /**
     * @param {Record<string, unknown>} model_inputs
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new YolosObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class YolosObjectDetectionOutput extends ModelOutput {
    logits;
    pred_boxes;
    /**
     * @param {Object} output The output of the model.
     * @param {Tensor} output.logits Classification logits (including no-object) for all queries.
     * @param {Tensor} output.pred_boxes Normalized boxes coordinates for all queries, represented as (center_x, center_y, width, height).
     * These values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding possible padding).
     */
    constructor({ logits, pred_boxes }: { logits: Tensor; pred_boxes: Tensor }) {
        super();
        this.logits = logits;
        this.pred_boxes = pred_boxes;
    }
}
