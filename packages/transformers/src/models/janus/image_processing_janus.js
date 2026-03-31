import { ImageProcessor } from '../../image_processors_utils.js';

export class VLMImageProcessor extends ImageProcessor {
    constant_values;
    /** @param {Record<string, unknown>} config */
    constructor(config) {
        super(/** @type {any} */ ({
            do_pad: true,
            pad_size: {
                width: /** @type {Record<string, number>} */ (config).image_size,
                height: /** @type {Record<string, number>} */ (config).image_size,
            },
            ...config,
        }));
        this.constant_values = /** @type {number[]} */ (/** @type {any} */ (this.config).background_color).map((/** @type {number} */ x) => x * this.rescale_factor);
    }

    /**
     * @param {Float32Array} pixelData
     * @param {number[]} imgDims
     * @param {{ width: number; height: number } | number | 'square'} padSize
     * @param {Record<string, unknown>} [options]
     */
    pad_image(pixelData, imgDims, padSize, options = {}) {
        return super.pad_image(pixelData, imgDims, padSize, {
            constant_values: this.constant_values,
            center: true,
            ...options,
        });
    }
}
