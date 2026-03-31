/**
 * @file Model registry for cache and file operations
 *
 * Provides static methods for:
 * - Discovering which files a model needs
 * - Detecting available quantization levels (dtypes)
 * - Getting file metadata
 * - Checking cache status
 *
 * **Example:** Get all files needed for a model
 * ```javascript
 * const files = await ModelRegistry.get_files(
 *   "onnx-community/all-MiniLM-L6-v2-ONNX",
 *   { dtype: "fp16" },
 * );
 * console.log(files); // [ 'config.json', 'onnx/model_fp16.onnx', 'onnx/model_fp16.onnx_data', 'tokenizer.json', 'tokenizer_config.json' ]
 * ```
 *
 * @module utils/model_registry
 */

import { clear_cache, clear_pipeline_cache } from './clear_cache.js';
import { get_available_dtypes } from './get_available_dtypes.js';
import { get_file_metadata } from './get_file_metadata.js';
import { get_files } from './get_files.js';
import { get_model_files } from './get_model_files.js';
import { get_pipeline_files } from './get_pipeline_files.js';
import { get_processor_files } from './get_processor_files.js';
import { get_tokenizer_files } from './get_tokenizer_files.js';
import { is_cached, is_cached_files, is_pipeline_cached, is_pipeline_cached_files } from './is_cached.js';

/**
 * Static class for cache and file management operations.
 * @hideconstructor
 */
export class ModelRegistry {
    /**
     * Get all files (model, tokenizer, processor) needed for a model.
     *
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @returns {Promise<string[]>} Array of file paths
     */
    static async get_files(modelId, options = {}) {
        return get_files(modelId, options);
    }

    /**
     * Get all files needed for a specific pipeline task.
     *
     * @param {string} task - The pipeline task
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @returns {Promise<string[]>} Array of file paths
     */
    static async get_pipeline_files(task, modelId, options = {}) {
        return get_pipeline_files(task, modelId, options);
    }

    /**
     * Get model files needed for a specific model.
     *
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @returns {Promise<string[]>} Array of model file paths
     */
    static async get_model_files(modelId, options = {}) {
        return get_model_files(modelId, options);
    }

    /**
     * Get tokenizer files needed for a specific model.
     *
     * @param {string} modelId - The model id
     * @returns {Promise<string[]>} Array of tokenizer file paths
     */
    static async get_tokenizer_files(modelId) {
        return get_tokenizer_files(modelId);
    }

    /**
     * Get processor files needed for a specific model.
     *
     * @param {string} modelId - The model id
     * @returns {Promise<string[]>} Array of processor file paths
     */
    static async get_processor_files(modelId) {
        return get_processor_files(modelId);
    }

    /**
     * Detects which quantization levels (dtypes) are available for a model.
     *
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @returns {Promise<string[]>} Array of available dtype strings
     */
    static async get_available_dtypes(modelId, options = {}) {
        return get_available_dtypes(modelId, options);
    }

    /**
     * Quickly checks if a model is fully cached.
     *
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @returns {Promise<boolean>} Whether all required files are cached
     */
    static async is_cached(modelId, options = {}) {
        return is_cached(modelId, options);
    }

    /**
     * Checks if all files for a given model are already cached, with per-file detail.
     *
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @returns {Promise<import('./is_cached.js').CacheCheckResult>} Object with allCached boolean and files array
     */
    static async is_cached_files(modelId, options = {}) {
        return is_cached_files(modelId, options);
    }

    /**
     * Quickly checks if all files for a specific pipeline task are cached.
     *
     * @param {string} task - The pipeline task
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @returns {Promise<boolean>} Whether all required files are cached
     */
    static async is_pipeline_cached(task, modelId, options = {}) {
        return is_pipeline_cached(task, modelId, options);
    }

    /**
     * Checks if all files for a specific pipeline task are already cached, with per-file detail.
     *
     * @param {string} task - The pipeline task
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @returns {Promise<import('./is_cached.js').CacheCheckResult>} Object with allCached boolean and files array
     */
    static async is_pipeline_cached_files(task, modelId, options = {}) {
        return is_pipeline_cached_files(task, modelId, options);
    }

    /**
     * Get metadata for a specific file without downloading it.
     *
     * @param {string} path_or_repo_id - Model id or path
     * @param {string} filename - The file name
     * @param {import('../hub.js').PretrainedOptions} [options] - Optional parameters
     * @returns {Promise<{exists: boolean, size?: number, contentType?: string, fromCache?: boolean}>} File metadata
     */
    static async get_file_metadata(path_or_repo_id, filename, options = {}) {
        return get_file_metadata(path_or_repo_id, filename, options);
    }

    /**
     * Clears all cached files for a given model.
     *
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @returns {Promise<import('./clear_cache.js').CacheClearResult>} Object with deletion statistics
     */
    static async clear_cache(modelId, options = {}) {
        return clear_cache(modelId, options);
    }

    /**
     * Clears all cached files for a specific pipeline task.
     *
     * @param {string} task - The pipeline task
     * @param {string} modelId - The model id
     * @param {Object} [options] - Optional parameters
     * @returns {Promise<import('./clear_cache.js').CacheClearResult>} Object with deletion statistics
     */
    static async clear_pipeline_cache(task, modelId, options = {}) {
        return clear_pipeline_cache(task, modelId, options);
    }
}
