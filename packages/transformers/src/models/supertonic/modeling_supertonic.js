import { Tensor, full, randn } from '../../utils/tensor.js';
import { PreTrainedModel } from '../modeling_utils.js';
import { sessionRun } from '../session.js';

export class SupertonicPreTrainedModel extends PreTrainedModel {}
export class SupertonicForConditionalGeneration extends SupertonicPreTrainedModel {
    /**
     * @param {Object} params
     * @param {Tensor} params.input_ids
     * @param {Tensor} params.attention_mask
     * @param {Tensor} params.style
     * @param {number} [params.num_inference_steps=5]
     * @param {number} [params.speed=1.05]
     */
    async generate_speech({
        // Required inputs
        input_ids,
        attention_mask,
        style,

        // Optional inputs
        num_inference_steps = 5,
        speed = 1.05,
    }) {
        const { sampling_rate, chunk_compress_factor, base_chunk_size, latent_dim } = /** @type {Record<string, number>} */ (/** @type {unknown} */ (this.config));

        // 1. Text Encoder
        const { last_hidden_state, durations: raw_durations } = await sessionRun(this.sessions['text_encoder'], {
            input_ids,
            attention_mask,
            style,
        });

        // Convert durations to sample counts
        const durations = /** @type {Tensor} */ (raw_durations).div(speed).mul_(sampling_rate);

        // 2. Latent Preparation
        // Calculate latent lengths: ceil(durations / latent_size)
        const latent_size = /** @type {number} */ (base_chunk_size) * /** @type {number} */ (chunk_compress_factor);
        const durationsData = /** @type {Float32Array} */ (durations.data);
        const latentLengths = Int32Array.from(durationsData, (d) => Math.ceil(d / latent_size));
        const maxLatentLen = Math.max(...latentLengths);

        // Create latent mask: (arange(max_len) < latent_lengths[:, None])
        const batch_size = input_ids.dims[0];
        const latentMaskData = new BigInt64Array(batch_size * maxLatentLen);
        for (let i = 0; i < batch_size; ++i) {
            latentMaskData.fill(1n, i * maxLatentLen, i * maxLatentLen + latentLengths[i]);
        }
        const latent_mask = new Tensor('int64', latentMaskData, [batch_size, maxLatentLen]);

        // Create initial noise and apply mask: latents *= latent_mask[:, None, :]
        const latentChannels = /** @type {number} */ (latent_dim) * /** @type {number} */ (chunk_compress_factor);
        const latentStride = latentChannels * maxLatentLen;
        let noisy_latents = randn([batch_size, latentChannels, maxLatentLen]);
        const latentsData = /** @type {Float32Array} */ (noisy_latents.data);
        for (let i = 0; i < batch_size; ++i) {
            if (latentLengths[i] === maxLatentLen) continue; // No padding for this item
            for (let c = 0; c < latentChannels; ++c) {
                latentsData.fill(
                    0,
                    i * latentStride + c * maxLatentLen + latentLengths[i],
                    i * latentStride + (c + 1) * maxLatentLen,
                );
            }
        }

        // 3. Denoising Loop
        const num_steps = full([batch_size], num_inference_steps);
        for (let step = 0; step < num_inference_steps; ++step) {
            const timestep = full([batch_size], step);
            ({ denoised_latents: noisy_latents } = await sessionRun(this.sessions['latent_denoiser'], {
                style,
                noisy_latents,
                latent_mask,
                encoder_outputs: /** @type {Tensor} */ (last_hidden_state),
                attention_mask,
                timestep,
                num_inference_steps: num_steps,
            }));
        }

        // 4. Voice Decoder
        const { waveform } = await sessionRun(this.sessions['voice_decoder'], {
            latents: noisy_latents,
        });

        return {
            waveform,
            durations,
        };
    }
}
