import { Processor } from '../../processing_utils';
import { AutoImageProcessor } from '../auto/image_processing_auto';
import { AutoTokenizer } from '../auto/tokenization_auto';
export class OwlViTProcessor extends Processor {
    static tokenizer_class = AutoTokenizer;
    static image_processor_class = AutoImageProcessor;
}
