import { ImageProcessor } from '../../image_processors_utils.js';

export class EfficientNetImageProcessor extends ImageProcessor {
    include_top;
    /** @param {Record<string, unknown>} config */
    constructor(config) {
        super(config);
        this.include_top = /** @type {any} */ (this.config).include_top ?? true;
        if (this.include_top) {
            this.image_std = /** @type {number[]} */ (this.image_std).map((/** @type {number} */ x) => x * x);
        }
    }
}
