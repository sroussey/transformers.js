import { ImageProcessor } from '../../image_processors_utils';

export class EfficientNetImageProcessor extends ImageProcessor {
    include_top;
    constructor(config) {
        super(config);
        this.include_top = this.config.include_top ?? true;
        if (this.include_top) {
            this.image_std = this.image_std.map((x) => x * x);
        }
    }
}
