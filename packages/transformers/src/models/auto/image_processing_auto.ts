import { ImageProcessor } from '../../image_processors_utils';
import { GITHUB_ISSUE_URL, IMAGE_PROCESSOR_NAME } from '../../utils/constants';
import { getModelJSON } from '../../utils/hub';
import { logger } from '../../utils/logger';
import * as AllImageProcessors from '../image_processors';

export class AutoImageProcessor {
    /** @type {typeof ImageProcessor.from_pretrained} */
    static async from_pretrained(pretrained_model_name_or_path: string, options: Record<string, unknown> = {}) {
        const preprocessorConfig = await getModelJSON(
            pretrained_model_name_or_path,
            IMAGE_PROCESSOR_NAME,
            true,
            options,
        );

        // Determine image processor class
        const key = (preprocessorConfig.image_processor_type ?? preprocessorConfig.feature_extractor_type) as string | undefined;
        let image_processor_class = (AllImageProcessors as unknown as Record<string, typeof ImageProcessor>)[key?.replace(/Fast$/, '') as string];

        if (!image_processor_class) {
            if (key !== undefined) {
                // Only log a warning if the class is not found and the key is set.
                logger.warn(
                    `Image processor type '${key}' not found, assuming base ImageProcessor. Please report this at ${GITHUB_ISSUE_URL}.`,
                );
            }
            image_processor_class = ImageProcessor;
        }

        // Instantiate image processor
        return new image_processor_class(preprocessorConfig);
    }
}
