import { ImageProcessor, post_process_semantic_segmentation } from '../../image_processors_utils';

export class SapiensImageProcessor extends ImageProcessor {
    /** @type {typeof post_process_semantic_segmentation} */
    post_process_semantic_segmentation(...args: Parameters<typeof post_process_semantic_segmentation>) {
        return post_process_semantic_segmentation(...(args as Parameters<typeof post_process_semantic_segmentation>));
    }
}
export class SapiensFeatureExtractor extends SapiensImageProcessor {}
