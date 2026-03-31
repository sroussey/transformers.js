import { Processor } from '../../processing_utils.js';
import { logger } from '../../utils/logger.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

const IMAGE_TOKEN = '<image>';

/**
 * @param {string} prompt
 * @param {string} bos_token
 * @param {number} image_seq_len
 * @param {string} image_token
 * @param {number} num_images
 */
function build_string_from_input(prompt, bos_token, image_seq_len, image_token, num_images) {
    return `${image_token.repeat(image_seq_len * num_images)}${bos_token}${prompt}\n`;
}

export class PaliGemmaProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;
    static uses_processor_config = false;

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
        if (!text) {
            logger.warn(
                'You are using PaliGemma without a text prefix. It will perform as a picture-captioning model.',
            );
            text = '';
        }

        if (!Array.isArray(images)) {
            images = [images];
        }

        if (!Array.isArray(text)) {
            text = [text];
        }

        const bos_token = /** @type {string} */ (/** @type {any} */ (this.tokenizer).bos_token);
        const image_seq_length = /** @type {number} */ (/** @type {any} */ (/** @type {any} */ (this.image_processor).config).image_seq_length);
        let input_strings;
        if (/** @type {string[]} */ (text).some((/** @type {string} */ t) => t.includes(IMAGE_TOKEN))) {
            input_strings = /** @type {string[]} */ (text).map((/** @type {string} */ sample) => {
                const expanded_sample = sample.replaceAll(IMAGE_TOKEN, IMAGE_TOKEN.repeat(image_seq_length));
                const bos_rfind_index = expanded_sample.lastIndexOf(IMAGE_TOKEN);
                const bos_index = bos_rfind_index === -1 ? 0 : bos_rfind_index + IMAGE_TOKEN.length;
                return expanded_sample.slice(0, bos_index) + bos_token + expanded_sample.slice(bos_index) + '\n';
            });
        } else {
            logger.warn(
                'You are passing both `text` and `images` to `PaliGemmaProcessor`. The processor expects special ' +
                    'image tokens in the text, as many tokens as there are images per each text. It is recommended to ' +
                    'add `<image>` tokens in the very beginning of your text. For this call, we will infer how many images ' +
                    'each text has and add special tokens.',
            );

            input_strings = /** @type {string[]} */ (text).map((/** @type {string} */ sample) =>
                build_string_from_input(sample, bos_token, image_seq_length, IMAGE_TOKEN, /** @type {import('../../utils/image.js').RawImage[]} */ (images).length),
            );
        }

        const text_inputs = /** @type {any} */ (this.tokenizer)(input_strings, kwargs);
        const image_inputs = await /** @type {any} */ (this.image_processor)(images, kwargs);

        return {
            ...image_inputs,
            ...text_inputs,
        };
    }
}
