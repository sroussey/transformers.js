import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

export class LlavaProcessor extends Processor {
    /** @type {Record<string, any>} */
    config = /** @type {any} */ (undefined);
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;
    static uses_processor_config = true;

    /**
     * @typedef {import('../../utils/image.js').RawImage} RawImage
     */

    // `images` is required, `text` is optional
    /**
     * @param {RawImage | RawImage[]} images
     * @param {string | string[] | null} [text]
     * @param {Record<string, any>} [kwargs]
     */
    async _call(images, text = null, kwargs = {}) {
        const image_inputs = await /** @type {any} */ (this.image_processor)(images, kwargs);

        if (text) {
            const [height, width] = /** @type {import('../../utils/tensor.js').Tensor} */ (image_inputs.pixel_values).dims.slice(-2);

            const { image_token, patch_size, num_additional_image_tokens } = this.config;
            const num_image_tokens =
                Math.floor(height / patch_size) * Math.floor(width / patch_size) + num_additional_image_tokens;

            text = structuredClone(text); // Avoid modifying the original text input
            if (!Array.isArray(text)) {
                text = [text];
            }
            for (let i = 0; i < text.length; ++i) {
                text[i] = text[i].replace(image_token, image_token.repeat(num_image_tokens));
            }
        }

        const text_inputs = text ? /** @type {any} */ (this.tokenizer)(text, kwargs) : {};

        return {
            ...image_inputs,
            ...text_inputs,
        };
    }
}
