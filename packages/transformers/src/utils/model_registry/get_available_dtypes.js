import { DATA_TYPES, DEFAULT_DTYPE_SUFFIX_MAPPING } from '../dtypes.js';
import { get_file_metadata } from './get_file_metadata.js';
import {
    MODEL_TYPES,
    MODEL_TYPE_MAPPING,
    MODEL_MAPPING_NAMES,
    getSessionsConfig,
} from '../../models/modeling_utils.js';
import { AutoConfig } from '../../configs.js';
import { GITHUB_ISSUE_URL } from '../constants.js';
import { logger } from '../logger.js';
import { memoizePromise } from '../memoize_promise.js';

/**
 * @typedef {import('../../configs.js').PretrainedConfig} PretrainedConfig
 */

/**
 * Returns a memoized AutoConfig for the given model ID and options.
 * @param {string} modelId
 * @param {Object} [options]
 * @param {PretrainedConfig|null} [options.config=null]
 * @param {string|null} [options.cache_dir=null]
 * @param {boolean} [options.local_files_only=false]
 * @param {string} [options.revision='main']
 * @returns {Promise<PretrainedConfig>}
 */
function get_config(modelId, { config = null, cache_dir = null, local_files_only = false, revision = 'main' } = {}) {
    if (config !== null) {
        return AutoConfig.from_pretrained(modelId, { config, cache_dir, local_files_only, revision });
    }
    const key = JSON.stringify([modelId, cache_dir, local_files_only, revision]);
    return memoizePromise(key, () =>
        AutoConfig.from_pretrained(modelId, { config, cache_dir, local_files_only, revision }),
    );
}

/**
 * The dtypes to probe for availability (excludes 'auto' which is not a concrete dtype).
 * @type {string[]}
 */
const CONCRETE_DTYPES = Object.keys(DEFAULT_DTYPE_SUFFIX_MAPPING);

/**
 * Detects which quantization levels (dtypes) are available for a model
 * by checking which ONNX files exist on the hub or locally.
 *
 * A dtype is considered available if *all* required model session files
 * exist for that dtype. For example, a Seq2Seq model needs both an encoder
 * and decoder file — the dtype is only listed if both are present.
 *
 * @param {string} modelId The model id (e.g., "onnx-community/all-MiniLM-L6-v2-ONNX")
 * @param {Object} [options] Optional parameters
 * @param {PretrainedConfig} [options.config=null] Pre-loaded model config (optional, will be fetched if not provided)
 * @param {string} [options.model_file_name=null] Override the model file name (excluding .onnx suffix)
 * @param {string} [options.revision='main'] Model revision
 * @param {string} [options.cache_dir=null] Custom cache directory
 * @param {boolean} [options.local_files_only=false] Only check local files
 * @returns {Promise<string[]>} Array of available dtype strings (e.g., ['fp32', 'fp16', 'q4', 'q8'])
 */
export async function get_available_dtypes(
    modelId,
    { config = null, model_file_name = null, revision = 'main', cache_dir = null, local_files_only = false } = {},
) {
    config = await get_config(modelId, { config, cache_dir, local_files_only, revision });

    const subfolder = 'onnx';

    // Determine model type (same logic as get_model_files)
    let modelType;
    const architectures = /** @type {string[]} */ (config.architectures || []);

    let foundInMapping = false;
    for (const arch of architectures) {
        const mappedType = MODEL_TYPE_MAPPING.get(arch);
        if (mappedType !== undefined) {
            modelType = mappedType;
            foundInMapping = true;
            break;
        }
    }

    if (!foundInMapping && config.model_type) {
        const mappedType = MODEL_TYPE_MAPPING.get(config.model_type);
        if (mappedType !== undefined) {
            modelType = mappedType;
            foundInMapping = true;
        }

        if (!foundInMapping) {
            for (const mapping of Object.values(MODEL_MAPPING_NAMES)) {
                if (mapping.has(config.model_type)) {
                    modelType = MODEL_TYPE_MAPPING.get(mapping.get(config.model_type));
                    foundInMapping = true;
                    break;
                }
            }
        }
    }

    if (!foundInMapping) {
        modelType = MODEL_TYPES.EncoderOnly;
    }

    const { sessions } = getSessionsConfig(modelType, config, { model_file_name });

    // Get all base names for model session files
    const baseNames = Object.values(sessions);

    // For each dtype, check if all session files exist
    const metadataOptions = { revision, cache_dir, local_files_only };

    // Probe all (dtype, baseName) combinations in parallel
    const probeResults = await Promise.all(
        CONCRETE_DTYPES.map(async (dtype) => {
            const suffix = DEFAULT_DTYPE_SUFFIX_MAPPING[dtype] ?? '';
            const allExist = await Promise.all(
                baseNames.map(async (baseName) => {
                    const filename = `${subfolder}/${baseName}${suffix}.onnx`;
                    const metadata = await get_file_metadata(modelId, filename, metadataOptions);
                    return metadata.exists;
                }),
            );
            return { dtype, available: allExist.every(Boolean) };
        }),
    );

    return probeResults.filter((r) => r.available).map((r) => r.dtype);
}
