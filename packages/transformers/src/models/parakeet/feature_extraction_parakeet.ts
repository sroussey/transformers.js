import { FeatureExtractor, validate_audio_inputs } from '../../feature_extraction_utils';
import { mel_filter_bank, spectrogram, window_function } from '../../utils/audio';
import { Tensor } from '../../utils/tensor';

const EPSILON = 1e-5;

export class ParakeetFeatureExtractor extends FeatureExtractor {
    window;
    constructor(config: Record<string, unknown>) {
        super(config);

        // Prefer given `mel_filters` from preprocessor_config.json, or calculate them if they don't exist.
        this.config.mel_filters ??= mel_filter_bank(
            Math.floor(1 + (this.config.n_fft as number) / 2), // num_frequency_bins
            this.config.feature_size as number, // num_mel_filters
            0.0, // min_frequency
            (this.config.sampling_rate as number) / 2, // max_frequency
            this.config.sampling_rate as number, // sampling_rate
            'slaney', // norm
            'slaney', // mel_scale
        );

        const window = window_function(this.config.win_length as number, 'hann', {
            periodic: false,
        });

        this.window = new Float64Array(this.config.n_fft as number);
        const offset = Math.floor(((this.config.n_fft as number) - (this.config.win_length as number)) / 2);
        this.window.set(window, offset);
    }

    /**
     * Computes the log-Mel spectrogram of the provided audio waveform.
     * @param {Float32Array|Float64Array} waveform The audio waveform to process.
     * @returns {Promise<Tensor>} An object containing the log-Mel spectrogram data as a Float32Array and its dimensions as an array of numbers.
     */
    async _extract_fbank_features(waveform: Float32Array | Float64Array) {
        // Parakeet uses a custom preemphasis strategy: Apply preemphasis to entire waveform at once
        const preemphasis = this.config.preemphasis as number;
        const waveformCopy = new Float64Array(waveform); // Clone to avoid destructive changes
        for (let j = waveformCopy.length - 1; j >= 1; --j) {
            waveformCopy[j] -= preemphasis * waveformCopy[j - 1];
        }

        const features = await spectrogram(
            waveformCopy,
            this.window, // window
            this.window.length, // frame_length
            this.config.hop_length as number, // hop_length
            {
                fft_length: this.config.n_fft as number,
                power: 2.0,
                mel_filters: this.config.mel_filters as number[][],
                log_mel: 'log',
                mel_floor: -Infinity,
                pad_mode: 'constant',
                center: true,

                // Custom
                transpose: true,
                mel_offset: 2 ** -24,
            },
        );

        return features;
    }

    /**
     * Asynchronously extracts features from a given audio using the provided configuration.
     * @param {Float32Array|Float64Array} audio The audio data as a Float32Array/Float64Array.
     * @returns {Promise<{ input_features: Tensor; attention_mask: Tensor; }>} A Promise resolving to an object containing the extracted input features as a Tensor.
     */
    async _call(audio: Float32Array | Float64Array) {
        validate_audio_inputs(audio, 'ParakeetFeatureExtractor');

        const features = await this._extract_fbank_features(audio);

        const n_fft = this.config.n_fft as number;
        const hop_length = this.config.hop_length as number;
        const features_length = Math.floor(
            (audio.length + Math.floor(n_fft / 2) * 2 - n_fft) / hop_length,
        );

        const features_data = features.data as Float32Array;
        features_data.fill(0, features_length * features.dims[1]);

        // normalize mel features, ignoring padding
        const [num_frames, num_features] = features.dims;
        const sum = new Float64Array(num_features);
        const sum_sq = new Float64Array(num_features);

        for (let i = 0; i < features_length; ++i) {
            const offset = i * num_features;
            for (let j = 0; j < num_features; ++j) {
                const val = features_data[offset + j];
                sum[j] += val;
                sum_sq[j] += val * val;
            }
        }

        // Calculate mean and standard deviation, then normalize
        const divisor = features_length > 1 ? features_length - 1 : 1;
        for (let j = 0; j < num_features; ++j) {
            const mean = sum[j] / features_length;
            const variance = (sum_sq[j] - features_length * mean * mean) / divisor;
            const std = Math.sqrt(variance) + EPSILON;
            const inv_std = 1 / std;

            for (let i = 0; i < features_length; ++i) {
                const index = i * num_features + j;
                features_data[index] = (features_data[index] - mean) * inv_std;
            }
        }

        const mask_data = new BigInt64Array(num_frames);
        mask_data.fill(1n, 0, features_length);

        return {
            input_features: features.unsqueeze_(0),
            attention_mask: new Tensor('int64', mask_data, [1, num_frames]),
        };
    }
}
