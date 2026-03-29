import { PreTrainedModel } from '../modeling_utils';
import { RTDetrObjectDetectionOutput } from '../rt_detr/modeling_rt_detr';

export class RFDetrPreTrainedModel extends PreTrainedModel {}
export class RFDetrModel extends RFDetrPreTrainedModel {}
export class RFDetrForObjectDetection extends RFDetrPreTrainedModel {
    /**
     * @param {Record<string, unknown>} model_inputs
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new RFDetrObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class RFDetrObjectDetectionOutput extends RTDetrObjectDetectionOutput {}
