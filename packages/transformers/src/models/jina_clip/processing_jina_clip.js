import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

export class JinaCLIPProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;

    /**
     * @param {string | string[] | null} [text]
     * @param {import('../../utils/image.js').RawImage | import('../../utils/image.js').RawImage[] | null} [images]
     * @param {Record<string, unknown>} [kwargs]
     */
    async _call(text = null, images = null, kwargs = {}) {
        if (!text && !images) {
            throw new Error('Either text or images must be provided');
        }

        const text_inputs = text ? /** @type {any} */ (this.tokenizer)(text, kwargs) : {};
        const image_inputs = images ? await /** @type {any} */ (this.image_processor)(images, kwargs) : {};

        return {
            ...text_inputs,
            ...image_inputs,
        };
    }
}
