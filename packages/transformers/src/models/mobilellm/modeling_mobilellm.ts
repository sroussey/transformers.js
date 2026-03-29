import { PreTrainedModel } from '../modeling_utils';

export class MobileLLMPreTrainedModel extends PreTrainedModel {}
export class MobileLLMModel extends MobileLLMPreTrainedModel {}
export class MobileLLMForCausalLM extends MobileLLMPreTrainedModel {}
