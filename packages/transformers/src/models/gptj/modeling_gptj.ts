import { PreTrainedModel } from '../modeling_utils';

export class GPTJPreTrainedModel extends PreTrainedModel {}
export class GPTJModel extends GPTJPreTrainedModel {}

export class GPTJForCausalLM extends GPTJPreTrainedModel {}
