import {
    ImageProcessor,
    post_process_instance_segmentation,
    post_process_panoptic_segmentation,
} from '../../image_processors_utils';

export class MaskFormerImageProcessor extends ImageProcessor {
    /** @type {typeof post_process_panoptic_segmentation} */
    post_process_panoptic_segmentation(...args: Parameters<typeof post_process_panoptic_segmentation>) {
        return post_process_panoptic_segmentation(...(args as Parameters<typeof post_process_panoptic_segmentation>));
    }
    /** @type {typeof post_process_instance_segmentation} */
    post_process_instance_segmentation(...args: Parameters<typeof post_process_instance_segmentation>) {
        return post_process_instance_segmentation(...(args as Parameters<typeof post_process_instance_segmentation>));
    }
}
export class MaskFormerFeatureExtractor extends MaskFormerImageProcessor {}
