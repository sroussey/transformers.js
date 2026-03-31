import { Processor } from '../../processing_utils.js';
import { AutoFeatureExtractor } from '../auto/feature_extraction_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

/**
 * Represents a ChatterboxProcessor that extracts features from an audio input.
 */
export class ChatterboxProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static feature_extractor_class = AutoFeatureExtractor;

    /**
     * @param {string | string[]} text
     * @param {Float32Array | Float64Array | null} [audio]
     */
    async _call(text, audio = null) {
        const text_features = /** @type {any} */ (this.tokenizer)(text);
        const audio_features = audio ? await /** @type {any} */ (this.feature_extractor)(audio) : {};
        return { ...text_features, ...audio_features };
    }
}
