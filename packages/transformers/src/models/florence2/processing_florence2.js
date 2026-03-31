import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';
import { AutoTokenizer } from '../auto/tokenization_auto.js';

export class Florence2Processor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;

    /** @type {Map<string, string>} */
    tasks_answer_post_processing_type;
    /** @type {Map<string, string>} */
    task_prompts_without_inputs;
    /** @type {Map<string, string>} */
    task_prompts_with_input;
    /** @type {{ quad_boxes: RegExp; bboxes: RegExp }} */
    regexes;
    /** @type {number} */
    size_per_bin;

    /**
     * @param {Record<string, unknown>} config
     * @param {Record<string, object>} components
     * @param {string | null} chat_template
     */
    constructor(config, components, chat_template) {
        super(config, components, chat_template);

        const {
            tasks_answer_post_processing_type,
            task_prompts_without_inputs,
            task_prompts_with_input,
        } = /** @type {any} */ (this.image_processor.config);

        /** @type {Map<string, string>} */
        this.tasks_answer_post_processing_type = new Map(Object.entries(tasks_answer_post_processing_type ?? {}));

        /** @type {Map<string, string>} */
        this.task_prompts_without_inputs = new Map(Object.entries(task_prompts_without_inputs ?? {}));

        /** @type {Map<string, string>} */
        this.task_prompts_with_input = new Map(Object.entries(task_prompts_with_input ?? {}));

        this.regexes = {
            quad_boxes:
                /(.+?)<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/gm,
            bboxes: /([^<]+)?<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>/gm,
        };
        this.size_per_bin = 1000;
    }

    /**
     * Helper function to construct prompts from input texts
     * @param {string|string[]} text
     * @returns {string[]}
     */
    construct_prompts(text) {
        if (typeof text === 'string') {
            text = [text];
        }

        const prompts = [];
        for (const t of text) {
            // 1. fixed task prompts without additional inputs
            if (this.task_prompts_without_inputs.has(t)) {
                prompts.push(this.task_prompts_without_inputs.get(t));
            }
            // 2. task prompts with additional inputs
            else {
                for (const [task, prompt] of this.task_prompts_with_input) {
                    if (t.includes(task)) {
                        prompts.push(prompt.replaceAll('{input}', t).replaceAll(task, ''));
                        break;
                    }
                }

                // 3. default prompt
                if (prompts.length !== text.length) {
                    prompts.push(t);
                }
            }
        }
        return prompts;
    }

    /**
     * Post-process the output of the model to each of the task outputs.
     * @param {string} text The text to post-process.
     * @param {string} task The task to post-process the text for.
     * @param {[number, number]} image_size The size of the image. height x width.
     */
    post_process_generation(text, task, image_size) {
        const task_answer_post_processing_type = this.tasks_answer_post_processing_type.get(task) ?? 'pure_text';

        // remove the special tokens
        text = text.replaceAll('<s>', '').replaceAll('</s>', '');

        let final_answer;
        switch (task_answer_post_processing_type) {
            case 'pure_text':
                final_answer = text;
                break;

            case 'description_with_bboxes':
            case 'bboxes':
            case 'phrase_grounding':
            case 'ocr':
                const key = task_answer_post_processing_type === 'ocr' ? 'quad_boxes' : 'bboxes';
                const matches = text.matchAll(this.regexes[key]);
                /** @type {string[]} */
                const labels = [];
                const items = [];
                for (const [_, label, ...locations] of matches) {
                    // Push new label, or duplicate the last label
                    labels.push(label ? label.trim() : (labels.at(-1) ?? ''));
                    items.push(
                        locations.map(
                            (/** @type {string} */ x, /** @type {number} */ i) =>
                                // NOTE: Add 0.5 to use the center position of the bin as the coordinate.
                                ((Number(x) + 0.5) / this.size_per_bin) * image_size[i % 2],
                        ),
                    );
                }
                final_answer = { labels, [key]: items };
                break;

            default:
                throw new Error(`Task "${task}" (of type "${task_answer_post_processing_type}") not yet implemented.`);
        }

        return { [task]: final_answer };
    }

    // NOTE: images and text are switched from the python version
    // `images` is required, `text` is optional
    /**
     * @param {unknown} images
     * @param {string | string[] | null} [text]
     * @param {Record<string, unknown>} [kwargs]
     */
    async _call(images, text = null, kwargs = {}) {
        if (!images && !text) {
            throw new Error('Either text or images must be provided');
        }

        const image_inputs = await /** @type {any} */ (this.image_processor)(images, kwargs);
        const text_inputs = text ? /** @type {any} */ (this.tokenizer)(this.construct_prompts(text), kwargs) : {};

        return {
            ...image_inputs,
            ...text_inputs,
        };
    }
}
