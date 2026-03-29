import { Processor } from '../../processing_utils';
import { RawImage } from '../../utils/image';
import { Tensor } from '../../utils/tensor';
import { AutoImageProcessor } from '../auto/image_processing_auto';
import { AutoTokenizer } from '../auto/tokenization_auto';

export class Qwen2VLProcessor extends Processor {
    declare config: Record<string, any>;
    static image_processor_class = AutoImageProcessor;
    static tokenizer_class = AutoTokenizer;
    static image_token = '<|image_pad|>';

    /**
     *
     * @param {string|string[]} text
     * @param {RawImage|RawImage[]} images
     * @param  {...any} args
     * @returns {Promise<any>}
     */
    async _call(text: string | string[], images: RawImage | RawImage[] | null = null, ...args: unknown[]) {
        if (!Array.isArray(text)) {
            text = [text];
        }

        let image_inputs, image_grid_thw;

        if (images) {
            image_inputs = await this.image_processor!(images);
            image_grid_thw = image_inputs.image_grid_thw;
        }

        if (image_grid_thw) {
            let merge_length = (this.image_processor!.config as Record<string, number>).merge_size ** 2;
            let index = 0;

            const image_token = (this.constructor as unknown as Record<string, string>).image_token;
            const image_grid_thw_list = (image_grid_thw as Tensor).tolist() as bigint[][];
            text = text.map((t: string) => {
                while (t.includes(image_token)) {
                    const prod = Number(image_grid_thw_list[index++].reduce((a: bigint, b: bigint) => a * b, 1n));
                    t = t.replace(image_token, '<|placeholder|>'.repeat(Math.floor(prod / merge_length)));
                }
                return t.replaceAll('<|placeholder|>', image_token);
            });
        }

        const text_inputs = this.tokenizer!(text);

        return {
            ...text_inputs,
            ...image_inputs,
        };
    }
}
