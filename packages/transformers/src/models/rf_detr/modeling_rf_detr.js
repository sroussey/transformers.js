import { PreTrainedModel } from '../modeling_utils.js';
import { RTDetrObjectDetectionOutput } from '../rt_detr/modeling_rt_detr.js';

export class RFDetrPreTrainedModel extends PreTrainedModel {}
export class RFDetrModel extends RFDetrPreTrainedModel {}
export class RFDetrForObjectDetection extends RFDetrPreTrainedModel {
    /**
     * @param {Record<string, unknown>} model_inputs
     */
    async _call(model_inputs) {
        return new RFDetrObjectDetectionOutput(/** @type {any} */ (await super._call(model_inputs)));
    }
}

export class RFDetrObjectDetectionOutput extends RTDetrObjectDetectionOutput {}
