import { PreTrainedModel } from '../modeling_utils';

export class GPTNeoXPreTrainedModel extends PreTrainedModel {}
export class GPTNeoXModel extends GPTNeoXPreTrainedModel {}

export class GPTNeoXForCausalLM extends GPTNeoXPreTrainedModel {}
