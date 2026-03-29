import { Processor } from '../../processing_utils';
import { AutoImageProcessor } from '../auto/image_processing_auto';
import { AutoTokenizer } from '../auto/tokenization_auto';

export class JinaCLIPProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;

    async _call(text: string | string[] | null = null, images: import('../../utils/image.js').RawImage | import('../../utils/image.js').RawImage[] | null = null, kwargs: Record<string, unknown> = {}) {
        if (!text && !images) {
            throw new Error('Either text or images must be provided');
        }

        const text_inputs = text ? this.tokenizer!(text, kwargs) : {};
        const image_inputs = images ? await this.image_processor!(images, kwargs) : {};

        return {
            ...text_inputs,
            ...image_inputs,
        };
    }
}
