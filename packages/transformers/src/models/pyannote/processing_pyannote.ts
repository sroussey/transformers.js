import { Processor } from '../../processing_utils';
import { PyAnnoteFeatureExtractor } from './feature_extraction_pyannote';

export class PyAnnoteProcessor extends Processor {
    static feature_extractor_class = PyAnnoteFeatureExtractor;

    /**
     * Calls the feature_extractor function with the given audio input.
     * @param {any} audio The audio input to extract features from.
     * @returns {Promise<any>} A Promise that resolves with the extracted features.
     */
    async _call(audio: Float32Array | Float64Array) {
        return await this.feature_extractor!(audio);
    }

    /** @type {PyAnnoteFeatureExtractor['post_process_speaker_diarization']} */
    post_process_speaker_diarization(...args: Parameters<PyAnnoteFeatureExtractor['post_process_speaker_diarization']>) {
        return (this.feature_extractor as unknown as PyAnnoteFeatureExtractor).post_process_speaker_diarization(
            ...args,
        );
    }

    get sampling_rate() {
        return this.feature_extractor.config.sampling_rate;
    }
}
