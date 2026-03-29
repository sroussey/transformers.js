import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils';
import { mel_filter_bank, spectrogram, window_function } from '../../utils/audio';
import { logger } from '../../utils/logger';
import { Tensor } from '../../utils/tensor';

export class WhisperFeatureExtractor extends FeatureExtractor {
    window;
    constructor(config: Record<string, unknown>) {
        super(config);

        // Prefer given `mel_filters` from preprocessor_config.json, or calculate them if they don't exist.
        this.config.mel_filters ??= mel_filter_bank(
            Math.floor(1 + (this.config.n_fft as number) / 2), // num_frequency_bins
            this.config.feature_size as number, // num_mel_filters
            0.0, // min_frequency
            8000.0, // max_frequency
            this.config.sampling_rate as number, // sampling_rate
            'slaney', // norm
            'slaney', // mel_scale
        );

        this.window = window_function(this.config.n_fft as number, 'hann');
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    async _extract_fbank_features(waveform: Float32Array | Float64Array) {
        return await spectrogram(
            waveform,
            this.window, // window
            this.config.n_fft as number, // frame_length
            this.config.hop_length as number, // hop_length
            {
                power: 2.0,
                mel_filters: this.config.mel_filters as number[][],
                log_mel: 'log10_max_norm',

                // Custom
                max_num_frames: Math.min(
                    Math.floor(waveform.length / (this.config.hop_length as number)),
                    this.config.nb_max_frames as number, // 3000
                ),
            },
        );
    }

    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_features: Tensor }>} A Promise resolving to an object containing the extracted input features as a Tensor.
     */
    async _call(audio: Float32Array | Float64Array, { max_length = null as number | null } = {}) {
        validate_audio_inputs(audio, 'WhisperFeatureExtractor');

        let waveform;
        const length = max_length ?? this.config.n_samples as number;
        if (audio.length > length) {
            if (audio.length > (this.config.n_samples as number)) {
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
