import { PreTrainedTokenizer } from '../../tokenization_utils';

export class LlamaTokenizer extends PreTrainedTokenizer {
    padding_side = 'left';
}
