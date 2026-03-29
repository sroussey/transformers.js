import { ImageProcessor } from '../../image_processors_utils';

export class VLMImageProcessor extends ImageProcessor {
    constant_values;
    constructor(config: Record<string, unknown>) {
        super({
            do_pad: true,
            pad_size: {
                width: (config as Record<string, number>).image_size,
                height: (config as Record<string, number>).image_size,
            },
            ...config,
        });
        this.constant_values = (this.config.background_color as number[]).map((x: number) => x * this.rescale_factor);
    }

    pad_image(pixelData: Float32Array, imgDims: number[], padSize: { width: number; height: number } | number | 'square', options: Record<string, unknown> = {}) {
        return super.pad_image(pixelData, imgDims, padSize, {
            constant_values: this.constant_values,
            center: true,
            ...options,
        });
    }
}
