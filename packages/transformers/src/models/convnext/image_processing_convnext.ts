import { ImageProcessor } from '../../image_processors_utils';

export class ConvNextImageProcessor extends ImageProcessor {
    crop_pct;
    constructor(config: Record<string, unknown>) {
        super(config);

        /**
         * Percentage of the image to crop. Only has an effect if this.size < 384.
         */
        this.crop_pct = this.config.crop_pct ?? 224 / 256;
    }

    async resize(image: import('../../utils/image.js').RawImage) {
        const shortest_edge = (this.size as Record<string, number>)?.shortest_edge;
        if (shortest_edge === undefined) {
            throw new Error(`Size dictionary must contain 'shortest_edge' key.`);
        }

        if (shortest_edge < 384) {
            // maintain same ratio, resizing shortest edge to shortest_edge/crop_pct
            const resize_shortest_edge = Math.floor(shortest_edge / (this.crop_pct as number));

            const [newWidth, newHeight] = this.get_resize_output_image_size(image, {
                shortest_edge: resize_shortest_edge,
            });

            image = await image.resize(newWidth, newHeight, {
                resample: this.resample as 0 | 1 | 2 | 3 | 4 | 5 | string,
            });

            // then crop to (shortest_edge, shortest_edge)
            image = await image.center_crop(shortest_edge, shortest_edge);
        } else {
            // warping (no cropping) when evaluated at 384 or larger
            image = await image.resize(shortest_edge, shortest_edge, {
                resample: this.resample as 0 | 1 | 2 | 3 | 4 | 5 | string,
            });
        }

        return image;
    }
}
export class ConvNextFeatureExtractor extends ConvNextImageProcessor {}
