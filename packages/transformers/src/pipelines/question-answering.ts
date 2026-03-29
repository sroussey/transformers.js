import { Pipeline, tensorAt } from './_base';

import { product } from '../utils/core';
import { softmax } from '../utils/maths';

/**
 * @typedef {import('./_base.js').TextPipelineConstructorArgs} TextPipelineConstructorArgs
 * @typedef {import('./_base.js').Disposable} Disposable
 */

/**
 * @typedef {Object} QuestionAnsweringOutput
 * @property {number} score The probability associated to the answer.
 * @property {number} [start] The character start index of the answer (in the tokenized version of the input).
 * @property {number} [end] The character end index of the answer (in the tokenized version of the input).
 * @property {string} answer The answer to the question.
 *
 * @typedef {Object} QuestionAnsweringPipelineOptions Parameters specific to question answering pipelines.
 * @property {number} [top_k=1] The number of top answer predictions to be returned.
 *
 * @callback QuestionAnsweringPipelineCallbackSingleTop1 Answer the question given as input by using the context.
 * @param {string} question The question.
 * @param {string} context The context.
 * @param {{top_k: 1} | {} | undefined} [options] The options to use for question answering.
 * @returns {Promise<QuestionAnsweringOutput>} The answer.
 *
 * @callback QuestionAnsweringPipelineCallbackSingleTopK Answer the question given as input by using the context.
 * @param {string} question The question.
 * @param {string} context The context.
 * @param {{top_k: number}} [options] The options to use for question answering.
 * @returns {Promise<QuestionAnsweringOutput[]>} The answers.
 *
 * @callback QuestionAnsweringPipelineCallbackBatchTop1 Answer the questions given as inputs by using the contexts.
 * @param {string[]} question The questions.
 * @param {string[]} context The contexts.
 * @param {{top_k: 1} | {} | undefined} [options] The options to use for question answering.
 * @returns {Promise<QuestionAnsweringOutput[]>} The answers.
 *
 * @callback QuestionAnsweringPipelineCallbackBatchTopK Answer the questions given as inputs by using the contexts.
 * @param {string[]} question The questions.
 * @param {string[]} context The contexts.
 * @param {{top_k: number}} [options] The options to use for question answering.
 * @returns {Promise<QuestionAnsweringOutput[][]>} The answers.
 *
 * @typedef {QuestionAnsweringPipelineCallbackSingleTop1 & QuestionAnsweringPipelineCallbackSingleTopK & QuestionAnsweringPipelineCallbackBatchTop1 & QuestionAnsweringPipelineCallbackBatchTopK} QuestionAnsweringPipelineCallback
 *
 * @typedef {TextPipelineConstructorArgs & QuestionAnsweringPipelineCallback & Disposable} QuestionAnsweringPipelineType
 */

/**
 * Question Answering pipeline using any `ModelForQuestionAnswering`.
 *
 * **Example:** Run question answering with `Xenova/distilbert-base-uncased-distilled-squad`.
 * ```javascript
 * import { pipeline } from '@huggingface/transformers';
 *
 * const answerer = await pipeline('question-answering', 'Xenova/distilbert-base-uncased-distilled-squad');
 * const question = 'Who was Jim Henson?';
 * const context = 'Jim Henson was a nice puppet.';
 * const output = await answerer(question, context);
 * // {
 * //   answer: "a nice puppet",
 * //   score: 0.5768911502526741
 * // }
 * ```
 */
export class QuestionAnsweringPipeline
    extends /** @type {new (options: TextPipelineConstructorArgs) => QuestionAnsweringPipelineType} */ (Pipeline)
{
    async _call(question: string | string[], context: string | string[], { top_k = 1 } = {}) {
        // Run tokenization
        const inputs = this.tokenizer(question, {
            text_pair: context,
            padding: true,
            truncation: true,
        });
        const isBatched = Array.isArray(question);

        const { start_logits, end_logits } = await this.model(inputs);
        const input_ids = inputs.input_ids.tolist() as bigint[][];
        // attention_mask[j] is compared loosely to 0 — it may be a sub-array (batched) or a number
        const attention_mask = inputs.attention_mask.tolist() as unknown[];

        // TODO: add support for `return_special_tokens_mask`
        const { all_special_ids, sep_token_id } = this.tokenizer;

        const batchedResults = [];
        for (let j = 0; j < start_logits.dims[0]; ++j) {
            const ids = input_ids[j];
            const sepIndex = ids.findIndex(
                (x: bigint) =>
                    // We use == to match bigint with number
                    x == (sep_token_id as unknown as bigint),
            );

            const start = tensorAt(start_logits, j).tolist() as number[];
            const end = tensorAt(end_logits, j).tolist() as number[];

            // Now, we mask out values that can't be in the answer
            // NOTE: We keep the cls_token unmasked (some models use it to indicate unanswerable questions)
            for (let i = 1; i < start.length; ++i) {
                if (
                    attention_mask[j] == 0 || // is part of padding
                    i <= sepIndex || // is before the sep_token
                    all_special_ids.findIndex((x: number) => x == Number(ids[i])) !== -1 // Is a special token
                ) {
                    // Make sure non-context indexes in the tensor cannot contribute to the softmax
                    start[i] = -Infinity;
                    end[i] = -Infinity;
                }
            }

            // Normalize logits and spans to retrieve the answer
            const start_scores = softmax(start).map((x: number, i: number) => [x, i]);
            const end_scores = softmax(end).map((x: number, i: number) => [x, i]);

            // Mask CLS
            start_scores[0][0] = 0;
            end_scores[0][0] = 0;

            // Generate all valid spans and select best ones
            const options = product(start_scores, end_scores)
                .filter((x: number[][]) => x[0][1] <= x[1][1])
                .map((x: number[][]) => [x[0][1], x[1][1], x[0][0] * x[1][0]])
                .sort((a: number[], b: number[]) => b[2] - a[2]);

            const sampleResults = [];
            for (let k = 0; k < Math.min(options.length, top_k); ++k) {
                const [start, end, score] = options[k];

                const answer_tokens = ids.slice(start, end + 1);

                const answer = this.tokenizer.decode(answer_tokens, {
                    skip_special_tokens: true,
                });

                // TODO add start and end?
                // NOTE: HF returns character index
                sampleResults.push({
                    answer,
                    score,
                });
            }
            if (top_k === 1) {
                batchedResults.push(...sampleResults);
            } else {
                batchedResults.push(sampleResults);
            }
        }

        return isBatched ? batchedResults : batchedResults[0];
    }
}
