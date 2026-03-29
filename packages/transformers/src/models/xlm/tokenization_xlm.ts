import { PreTrainedTokenizer } from '../../tokenization_utils';
import { logger } from '../../utils/logger';

export class XLMTokenizer extends PreTrainedTokenizer {
    return_token_type_ids = true;

    constructor(tokenizerJSON: Record<string, any>, tokenizerConfig: Record<string, any>) {
        super(tokenizerJSON, tokenizerConfig);
        logger.warn(
            'WARNING: `XLMTokenizer` is not yet supported by Hugging Face\'s "fast" tokenizers library. Therefore, you may experience slightly inaccurate results.',
        );
    }
}
