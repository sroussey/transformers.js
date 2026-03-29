import { PreTrainedModel } from '../modeling_utils';
import { RTDetrObjectDetectionOutput } from '../rt_detr/modeling_rt_detr';

export class DFinePreTrainedModel extends PreTrainedModel {}
export class DFineModel extends DFinePreTrainedModel {}
export class DFineForObjectDetection extends DFinePreTrainedModel {
    /**
     * @param {Object} model_inputs
     */
    async _call(model_inputs: Record<string, unknown>) {
        return new RTDetrObjectDetectionOutput(await super._call(model_inputs));
    }
}
