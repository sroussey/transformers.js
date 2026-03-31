import { Processor } from '../../processing_utils.js';
import { AutoImageProcessor } from '../auto/image_processing_auto.js';

export class SamProcessor extends Processor {
    static image_processor_class = AutoImageProcessor;

    /** @param {any[]} args */
    async _call(...args) {
        return await /** @type {any} */ (this.image_processor)(...args);
    }

    /** @type {import('./image_processing_sam.js').SamImageProcessor['post_process_masks']} */
    post_process_masks(...args) {
        return /** @type {import('./image_processing_sam.js').SamImageProcessor} */ (/** @type {unknown} */ (this.image_processor)).post_process_masks(...args);
    }

    /** @type {import('./image_processing_sam.js').SamImageProcessor['reshape_input_points']} */
    reshape_input_points(...args) {
        return /** @type {import('./image_processing_sam.js').SamImageProcessor} */ (/** @type {unknown} */ (this.image_processor)).reshape_input_points(...args);
    }
}
