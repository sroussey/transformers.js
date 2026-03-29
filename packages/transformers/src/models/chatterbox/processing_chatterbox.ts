import { Processor } from '../../processing_utils';
import { AutoFeatureExtractor } from '../auto/feature_extraction_auto';
import { AutoTokenizer } from '../auto/tokenization_auto';

/**
 * Represents a ChatterboxProcessor that extracts features from an audio input.
 */
export class ChatterboxProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static feature_extractor_class = AutoFeatureExtractor;

    async _call(text, audio = null) {
        const text_features = this.tokenizer(text);
        const audio_features = audio ? await this.feature_extractor(audio) : {};
        return { ...text_features, ...audio_features };
    }
}
