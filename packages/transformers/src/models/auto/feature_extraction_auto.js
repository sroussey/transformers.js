import { FeatureExtractor } from '../../feature_extraction_utils.js';
import { FEATURE_EXTRACTOR_NAME, GITHUB_ISSUE_URL } from '../../utils/constants.js';
import { getModelJSON } from '../../utils/hub.js';
import * as AllFeatureExtractors from '../feature_extractors.js';

export class AutoFeatureExtractor {
    /** @type {typeof FeatureExtractor.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path, options = {}) {
        const preprocessorConfig = await getModelJSON(
            pretrained_model_name_or_path,
            FEATURE_EXTRACTOR_NAME,
            true,
            options,
        );

        // Determine feature extractor class
        const key = /** @type {string} */ (/** @type {any} */ (preprocessorConfig).feature_extractor_type);
        const feature_extractor_class = /** @type {Record<string, typeof FeatureExtractor>} */ (/** @type {unknown} */ (AllFeatureExtractors))[key];

        if (!feature_extractor_class) {
            throw new Error(`Unknown feature_extractor_type: '${key}'. Please report this at ${GITHUB_ISSUE_URL}.`);
        }

        // Instantiate feature extractor
        return new feature_extractor_class(preprocessorConfig);
    }
}
