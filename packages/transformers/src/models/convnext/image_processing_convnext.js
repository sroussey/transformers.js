import { ImageProcessor } from '../../image_processors_utils.js';

export class ConvNextImageProcessor extends ImageProcessor {
    crop_pct;
    /** @param {Record<string, unknown>} config */
    constructor(config) {
        super(config);

        /**
         * Percentage of the image to crop. Only has an effect if this.size < 384.
         */
        this.crop_pct = /** @type {any} */ (this.config).crop_pct ?? 224 / 256;
    }

    /** @param {import('../../utils/image.js').RawImage} image */
    async resize(image) {
        const shortest_edge = /** @type {Record<string, number>} */ (this.size)?.shortest_edge;
        if (shortest_edge === undefined) {
            throw new Error(`Size dictionary must contain 'shortest_edge' key.`);
        }

        if (shortest_edge < 384) {
            // maintain same ratio, resizing shortest edge to shortest_edge/crop_pct
            const resize_shortest_edge = Math.floor(shortest_edge / /** @type {number} */ (this.crop_pct));

            const [newWidth, newHeight] = this.get_resize_output_image_size(image, {
                shortest_edge: resize_shortest_edge,
            });

            image = await image.resize(newWidth, newHeight, {
                resample: /** @type {0 | 1 | 2 | 3 | 4 | 5 | string} */ (this.resample),
            });

            // then crop to (shortest_edge, shortest_edge)
            image = await image.center_crop(shortest_edge, shortest_edge);
        } else {
            // warping (no cropping) when evaluated at 384 or larger
            image = await image.resize(shortest_edge, shortest_edge, {
                resample: /** @type {0 | 1 | 2 | 3 | 4 | 5 | string} */ (this.resample),
            });
        }

        return image;
    }
}
export class ConvNextFeatureExtractor extends ConvNextImageProcessor {}
