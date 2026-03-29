import { PreTrainedModel } from '../modeling_utils';

export class MaskFormerPreTrainedModel extends PreTrainedModel {}
export class MaskFormerModel extends MaskFormerPreTrainedModel {}
export class MaskFormerForInstanceSegmentation extends MaskFormerPreTrainedModel {}
