import { ImageProcessor, post_process_semantic_segmentation } from '../../image_processors_utils';

export class SegformerImageProcessor extends ImageProcessor {
    /** @type {typeof post_process_semantic_segmentation} */
    post_process_semantic_segmentation(...args) {
        // @ts-ignore
        return post_process_semantic_segmentation(...args);
    }
}
export class SegformerFeatureExtractor extends SegformerImageProcessor {}
