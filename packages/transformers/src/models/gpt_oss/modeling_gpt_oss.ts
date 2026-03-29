import { PreTrainedModel } from '../modeling_utils';

export class GptOssPreTrainedModel extends PreTrainedModel {}
export class GptOssModel extends GptOssPreTrainedModel {}
export class GptOssForCausalLM extends GptOssPreTrainedModel {}
