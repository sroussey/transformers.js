import { PreTrainedModel } from '../modeling_utils';
import { RTDetrObjectDetectionOutput } from '../rt_detr/modeling_rt_detr';

export class RTDetrV2PreTrainedModel extends PreTrainedModel {}
export class RTDetrV2Model extends RTDetrV2PreTrainedModel {}
export class RTDetrV2ForObjectDetection extends RTDetrV2PreTrainedModel {
    async _call(model_inputs: Record<string, any>) {
        return new RTDetrV2ObjectDetectionOutput(await super._call(model_inputs));
    }
}

export class RTDetrV2ObjectDetectionOutput extends RTDetrObjectDetectionOutput {}
