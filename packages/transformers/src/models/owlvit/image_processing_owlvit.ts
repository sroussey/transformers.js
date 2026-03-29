import { ImageProcessor, post_process_object_detection } from '../../image_processors_utils';

export class OwlViTImageProcessor extends ImageProcessor {
    /** @type {typeof post_process_object_detection} */
    post_process_object_detection(...args: Parameters<typeof post_process_object_detection>) {
        return post_process_object_detection(...args);
    }
}
export class OwlViTFeatureExtractor extends OwlViTImageProcessor {}
