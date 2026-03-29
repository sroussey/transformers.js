import { PreTrainedModel } from '../modeling_utils';

export class YoutuPreTrainedModel extends PreTrainedModel {}
export class YoutuModel extends YoutuPreTrainedModel {}
export class YoutuForCausalLM extends YoutuPreTrainedModel {}
