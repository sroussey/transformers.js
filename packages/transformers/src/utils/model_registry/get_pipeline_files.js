import { SUPPORTED_TASKS, TASK_ALIASES } from '../../pipelines/index.js';
import { get_files } from './get_files.js';

/**
 * Get all files needed for a specific pipeline task.
 *
 * @param {string} task - The pipeline task (e.g., "text-generation", "image-classification")
 * @param {string} modelId - The model id (e.g., "Xenova/bert-base-uncased")
 * @param {Object} [options] - Optional parameters
 * @param {import('../../configs.js').PretrainedConfig} [options.config=null] - Pre-loaded config
 * @param {import('../dtypes.js').DataType|Record<string, import('../dtypes.js').DataType>} [options.dtype=null] - Override dtype
 * @param {import('../devices.js').DeviceType|Record<string, import('../devices.js').DeviceType>} [options.device=null] - Override device
 * @param {string} [options.model_file_name=null] - Override the model file name (excluding .onnx suffix)
 * @returns {Promise<string[]>} Array of file paths that will be loaded
 * @throws {Error} If the task is not supported
 */
export async function get_pipeline_files(task, modelId, options = {}) {
    // Apply task aliases
    task = /** @type {Record<string, string>} */ (TASK_ALIASES)[task] ?? task;

    // Validate that the task is supported
    const taskConfig = /** @type {Record<string, { pipeline: unknown, model: unknown, default: unknown, type: string }>} */ (SUPPORTED_TASKS)[task];
    if (!taskConfig) {
        throw new Error(
            `Unsupported pipeline task: ${task}. Must be one of [${Object.keys(SUPPORTED_TASKS).join(', ')}]`,
        );
    }

    const { type } = taskConfig;
    const include_tokenizer = type !== 'audio' && type !== 'image';
    const include_processor = type !== 'text';

    return get_files(modelId, {
        ...options,
        include_tokenizer,
        include_processor,
    });
}
