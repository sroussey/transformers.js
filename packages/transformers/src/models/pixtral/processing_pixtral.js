import { Processor } from '../../processing_utils.js';
import { Tensor } from '../../utils/tensor.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

export class PixtralProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;
    static uses_processor_config = true;

    /**
     * 
     */

    // `images` is required, `text` is optional
    /**
     * @param {import('../../utils/image.js').RawImage|import('../../utils/image.js').RawImage[]} images
     * @param {string | string[] | null} [text=null]
     * @param {Record<string, unknown>} [kwargs={}]
     */
    async _call(images, text = null, kwargs = {}) {
        const image_inputs = await /** @type {any} */ (this.image_processor)(images, kwargs);

        if (text) {
            const [height, width] = /** @type {Tensor} */ (image_inputs.pixel_values).dims.slice(-2);

            const { image_token, image_break_token, image_end_token, patch_size, spatial_merge_size } = /** @type {Record<string, any>} */ (this.config);
            const real_patch_size = /** @type {number} */ (patch_size) * /** @type {number} */ (spatial_merge_size);
            const num_height_tokens = Math.floor(height / real_patch_size);
            const num_width_tokens = Math.floor(width / real_patch_size);

            text = structuredClone(text); // Avoid modifying the original text input
            if (!Array.isArray(text)) {
                text = [text];
            }
            for (let i = 0; i < text.length; ++i) {
                const width_tokens = /** @type {string} */ (image_token).repeat(num_width_tokens);
                const row = width_tokens + image_break_token;
                const finalRow = width_tokens + image_end_token;
                const full = row.repeat(num_height_tokens - 1) + finalRow;
                text[i] = text[i].replace(image_token, full);
            }
        }

        const text_inputs = text ? /** @type {any} */ (this.tokenizer)(text, kwargs) : {};

        return {
            ...image_inputs,
            ...text_inputs,
        };
    }
}
