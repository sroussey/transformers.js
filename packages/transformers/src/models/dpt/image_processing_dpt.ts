import { ImageProcessor } from '../../image_processors_utils';

export class DPTImageProcessor extends ImageProcessor {}
export class DPTFeatureExtractor extends DPTImageProcessor {} // NOTE: extends DPTImageProcessor
