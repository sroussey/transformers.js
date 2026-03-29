import { Pipeline, prepareImages } from './_base';

import { softmax } from '../utils/maths';
import { Tensor, topk } from '../utils/tensor';

/**
 * @typedef {import('./_base.js').ImagePipelineConstructorArgs} ImagePipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 * @typedef {import('./_base.js').ImageInput} ImageInput
 */

/**
 * @typedef {Object} ImageClassificationSingle
 * @property {string} label The label identified by the model.
 * @property {number} score The score attributed by the model for that label.
 * @typedef {ImageClassificationSingle[]} ImageClassificationOutput
 *
 * @typedef {Object} ImageClassificationPipelineOptions Parameters specific to image classification pipelines.
 * @property {number} [top_k=1] The number of top labels that will be returned by the pipeline.
 *
 * @callback ImageClassificationPipelineCallbackSingle Assign labels to the image passed as input.
 * @param {ImageInput} images The input image to be classified.
 * @param {ImageClassificationPipelineOptions} [options] The options to use for image classification.
 * @returns {Promise<ImageClassificationOutput>} An array containing the predicted labels and scores.
 *
 * @callback ImageClassificationPipelineCallbackBatch Assign labels to the images passed as inputs.
 * @param {ImageInput[]} images The input images to be classified.
 * @param {ImageClassificationPipelineOptions} [options] The options to use for image classification.
 * @returns {Promise<ImageClassificationOutput[]>} An array where each entry contains the predictions for the corresponding input image.
 *
 * @typedef {ImageClassificationPipelineCallbackSingle & ImageClassificationPipelineCallbackBatch} ImageClassificationPipelineCallback
 *
 * @typedef {ImagePipelineConstructorArgs & ImageClassificationPipelineCallback & Disposable} ImageClassificationPipelineType
 */

/**
 * Image classification pipeline using any `AutoModelForImageClassification`.
 * This pipeline predicts the class of an image.
 *
 * **Example:** Classify an image.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
 * const output = await classifier(url);
 * // [
 * //   { label: 'tiger, Panthera tigris', score: 0.632695734500885 },
 * // ]
 * ```
 *
 * **Example:** Classify an image and return top `n` classes.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
 * const output = await classifier(url, { top_k: 3 });
 * // [
 * //   { label: 'tiger, Panthera tigris', score: 0.632695734500885 },
 * //   { label: 'tiger cat', score: 0.3634825646877289 },
 * //   { label: 'lion, king of beasts, Panthera leo', score: 0.00045060308184474707 },
 * // ]
 * ```
 *
 * **Example:** Classify an image and return all classes.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const classifier = await pipeline('image-classification', 'Xenova/vit-base-patch16-224');
 * const url = 'https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/tiger.jpg';
 * const output = await classifier(url, { top_k: 0 });
 * // [
 * //   { label: 'tiger, Panthera tigris', score: 0.632695734500885 },
 * //   { label: 'tiger cat', score: 0.3634825646877289 },
 * //   { label: 'lion, king of beasts, Panthera leo', score: 0.00045060308184474707 },
 * //   { label: 'jaguar, panther, Panthera onca, Felis onca', score: 0.00035465499968267977 },
 * //   ...
 * // ]
 * ```
 */
export class ImageClassificationPipeline
    extends /** @type {new (options: ImagePipelineConstructorArgs) => ImageClassificationPipelineType} */ (Pipeline)
{
    async _call(images: import('./_base.js').ImageInput | import('./_base.js').ImageInput[], { top_k = 5 } = {}) {
        const preparedImages = await prepareImages(images);

        const { pixel_values } = await this.processor(preparedImages);
        const output = await this.model({ pixel_values });

        const { id2label } = this.model.config;

        /** @type {ImageClassificationOutput[]} */
        const toReturn = [];
        for (const batch_ of output.logits) {
            const batch = batch_ as Tensor;
            const scores = await topk(new Tensor('float32', softmax(batch.data as Float32Array), batch.dims), top_k) as Tensor[];

            const values = scores[0].tolist() as number[];
            const indices = scores[1].tolist() as number[];

            const vals = indices.map((x: number, i: number) => ({
                label: /** @type {string} */ (id2label ? (id2label as Record<number, string>)[x] : `LABEL_${x}`),
                score: /** @type {number} */ (values[i]),
            }));
            toReturn.push(vals);
        }

        return Array.isArray(images) ? toReturn : toReturn[0];
    }
}
