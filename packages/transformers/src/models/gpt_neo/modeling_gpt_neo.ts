import { PreTrainedModel } from '../modeling_utils';

export class GPTNeoPreTrainedModel extends PreTrainedModel {}
export class GPTNeoModel extends GPTNeoPreTrainedModel {}

export class GPTNeoForCausalLM extends GPTNeoPreTrainedModel {}
