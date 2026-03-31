import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils.js';
import { mel_filter_bank, spectrogram, window_function } from '../../utils/audio.js';
import { logger } from '../../utils/logger.js';
import { Tensor } from '../../utils/tensor.js';

export class WhisperFeatureExtractor extends FeatureExtractor {
    window;
    /** @param {Record<string, unknown>} config */
    constructor(config) {
        super(config);
        const cfg = /** @type {Record<string, any>} */ (this.config);

        // Prefer given `mel_filters` from preprocessor_config.json, or calculate them if they don't exist.
        cfg.mel_filters ??= mel_filter_bank(
            Math.floor(1 + /** @type {number} */ (cfg.n_fft) / 2), // num_frequency_bins
            /** @type {number} */ (cfg.feature_size), // num_mel_filters
            0.0, // min_frequency
            8000.0, // max_frequency
            /** @type {number} */ (cfg.sampling_rate), // sampling_rate
            'slaney', // norm
            'slaney', // mel_scale
        );

        this.window = window_function(/** @type {number} */ (cfg.n_fft), 'hann');
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    async _extract_fbank_features(waveform) {
        const cfg = /** @type {Record<string, any>} */ (this.config);
        return await spectrogram(
            waveform,
            this.window, // window
            /** @type {number} */ (cfg.n_fft), // frame_length
            /** @type {number} */ (cfg.hop_length), // hop_length
            {
                power: 2.0,
                mel_filters: /** @type {number[][]} */ (cfg.mel_filters),
                log_mel: 'log10_max_norm',

                // Custom
                max_num_frames: Math.min(
                    Math.floor(waveform.length / /** @type {number} */ (cfg.hop_length)),
                    /** @type {number} */ (cfg.nb_max_frames), // 3000
                ),
            },
        );
    }

    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_features: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
     */
    async _call(audio, { max_length = /** @type {number | null} */ (null) } = {}) {
        validate_audio_inputs(audio, 'WhisperFeatureExtractor');

        let waveform;
        const cfg = /** @type {Record<string, any>} */ (this.config);
        const length = max_length ?? /** @type {number} */ (cfg.n_samples);
        if (audio.length > length) {
            if (audio.length > /** @type {number} */ (cfg.n_samples)) {
                logger.warn(
                    'Attempting to extract features for audio longer than 30 seconds. ' +
                        'If using a pipeline to extract transcript from a long audio clip, ' +
                        'remember to specify `chunk_length_s` and/or `stride_length_s`.',
                );
            }
            waveform = audio.slice(0, length);
        } else {
            // pad with zeros
            waveform = new Float32Array(length);
            waveform.set(audio);
        }

        const features = await this._extract_fbank_features(waveform);

        return {
            input_features: features.unsqueeze_(0),
        };
    }
}
