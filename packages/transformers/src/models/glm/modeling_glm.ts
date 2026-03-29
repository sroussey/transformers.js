import { PreTrainedModel } from '../modeling_utils';

export class GlmPreTrainedModel extends PreTrainedModel {}
export class GlmModel extends GlmPreTrainedModel {}
export class GlmForCausalLM extends GlmPreTrainedModel {}
