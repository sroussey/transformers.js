import { Decoder } from '@huggingface/tokenizers';

import { PreTrainedTokenizer } from '../../tokenization_utils';

class VitsDecoder extends Decoder {
    /** @type {Decoder['decode_chain']} */
    decode_chain(tokens: string[]) {
        let decoded = '';
        for (let i = 1; i < tokens.length; i += 2) {
            decoded += tokens[i];
        }
        return [decoded];
    }
}
export class VitsTokenizer extends PreTrainedTokenizer {
    constructor(tokenizerJSON: Record<string, unknown>, tokenizerConfig: Record<string, unknown>) {
        super(tokenizerJSON, tokenizerConfig);

        // Custom decoder function
        this._tokenizer.decoder = new VitsDecoder({ type: 'VitsDecoder' });
    }
}
