import { ImageProcessor } from '../../image_processors_utils.js';

export class VLMImageProcessor extends ImageProcessor {
    constant_values;
    constructor(config) {
        super({
            do_pad: true,
            pad_size: {
                width: config.image_size,
                height: config.image_size,
            },
            ...config,
        });
        this.constant_values = this.config.background_color.map((x) => x * this.rescale_factor);
    }

    pad_image(pixelData, imgDims, padSize, options) {
        return super.pad_image(pixelData, imgDims, padSize, {
            constant_values: this.constant_values,
            center: true,
            ...options,
        });
    }
}
