import { PreTrainedTokenizer, _build_translation_inputs } from '../../tokenization_utils';

export class MBartTokenizer extends PreTrainedTokenizer {
    languageRegex;
    language_codes;
    lang_to_token;
    constructor(tokenizerJSON: Record<string, unknown>, tokenizerConfig: Record<string, unknown>) {
        super(tokenizerJSON, tokenizerConfig);

        this.languageRegex = /^[a-z]{2}_[A-Z]{2}$/;
        this.language_codes = this.all_special_tokens.filter((x: string) => this.languageRegex.test(x)).map((x: string) => x);
        this.lang_to_token = (x: string) => x; // Identity function
    }

    /**
     * Helper function to build translation inputs for an `MBartTokenizer`.
     * @param {string|string[]} raw_inputs The text to tokenize.
     * @param {Object} tokenizer_options Options to be sent to the tokenizer
     * @param {Object} generate_kwargs Generation options.
     * @returns {Object} Object to be passed to the model.
     */
    _build_translation_inputs(raw_inputs: string | string[], tokenizer_options: Record<string, unknown>, generate_kwargs: Record<string, unknown>) {
        return _build_translation_inputs(this, raw_inputs, tokenizer_options, generate_kwargs);
    }
}
