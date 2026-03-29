import { FeatureExtractor } from '../../feature_extraction_utils';
import { FEATURE_EXTRACTOR_NAME, GITHUB_ISSUE_URL } from '../../utils/constants';
import { getModelJSON } from '../../utils/hub';
import * as AllFeatureExtractors from '../feature_extractors';

export class AutoFeatureExtractor {
    /** @type {typeof FeatureExtractor.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path: string, options: Record<string, unknown> = {}) {
        const preprocessorConfig = await getModelJSON(
            pretrained_model_name_or_path,
            FEATURE_EXTRACTOR_NAME,
            true,
            options,
        );

        // Determine feature extractor class
        const key = preprocessorConfig.feature_extractor_type as string;
        const feature_extractor_class = (AllFeatureExtractors as unknown as Record<string, typeof FeatureExtractor>)[key];

        if (!feature_extractor_class) {
            throw new Error(`Unknown feature_extractor_type: '${key}'. Please report this at ${GITHUB_ISSUE_URL}.`);
        }

        // Instantiate feature extractor
        return new feature_extractor_class(preprocessorConfig);
    }
}
