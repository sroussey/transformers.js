import { Processor } from '../../processing_utils';
import { AutoImageProcessor } from '../auto/image_processing_auto';

export class SamProcessor extends Processor {
    static image_processor_class = AutoImageProcessor;

    async _call(...args: [images: unknown, options?: Record<string, unknown>]) {
        return await this.image_processor!(...args);
    }

    post_process_masks(...args: Parameters<import('./image_processing_sam').SamImageProcessor['post_process_masks']>) {
        return (this.image_processor as unknown as import('./image_processing_sam').SamImageProcessor).post_process_masks(...args);
    }

    reshape_input_points(...args: Parameters<import('./image_processing_sam').SamImageProcessor['reshape_input_points']>) {
        return (this.image_processor as unknown as import('./image_processing_sam').SamImageProcessor).reshape_input_points(...args);
    }
}
