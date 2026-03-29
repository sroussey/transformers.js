import { Processor } from '../../processing_utils';
import { RawAudio } from '../../utils/audio';
import { RawImage } from '../../utils/image';
import { AutoFeatureExtractor } from '../auto/feature_extraction_auto';
import { AutoImageProcessor } from '../auto/image_processing_auto';
import { AutoTokenizer } from '../auto/tokenization_auto';

export class Gemma3nProcessor extends Processor {
    declare config: Record<string, any>;
    static image_processor_class = AutoImageProcessor;
    static feature_extractor_class = AutoFeatureExtractor;
    static tokenizer_class = AutoTokenizer;
    static uses_processor_config = true;
    static uses_chat_template_file = true;

    audio_seq_length: number;
    image_seq_length: number;
    audio_token_id: string;
    boa_token: string;
    audio_token: string;
    full_audio_sequence: string;
    image_token_id: string;
    boi_token: string;
    image_token: string;
    full_image_sequence: string;
    constructor(config: Record<string, any>, components: Record<string, object>, chat_template: string | null) {
        super(config, components, chat_template);
        this.audio_seq_length = this.config.audio_seq_length;
        this.image_seq_length = this.config.image_seq_length;

        const {
            // Audio tokens
            audio_token_id,
            boa_token,
            audio_token,
            eoa_token,

            // Image tokens
            image_token_id,
            boi_token,
            image_token,
            eoi_token,
        } = this.tokenizer!.config as Record<string, string>;

        this.audio_token_id = audio_token_id;
        this.boa_token = boa_token;
        this.audio_token = audio_token;
        const audio_tokens_expanded = this.audio_token.repeat(this.audio_seq_length);
        this.full_audio_sequence = `\n\n${boa_token}${audio_tokens_expanded}${eoa_token}\n\n`;

        this.image_token_id = image_token_id;
        this.boi_token = boi_token;
        this.image_token = image_token;
        const image_tokens_expanded = (image_token as string).repeat(this.image_seq_length);
        this.full_image_sequence = `\n\n${boi_token}${image_tokens_expanded}${eoi_token}\n\n`;
    }

    /**
     *
     * @param {string|string[]} text
     * @param {RawImage|RawImage[]|RawImage[][]} images
     * @param {RawAudio|RawAudio[]|RawAudio[][]} audio
     * @returns {Promise<any>}
     */
    async _call(text: string | string[], images: RawImage | RawImage[] | RawImage[][] | null = null, audio: RawAudio | RawAudio[] | RawAudio[][] | null = null, options = {}) {
        if (typeof text === 'string') {
            text = [text];
        }

        let audio_inputs;
        if (audio) {
            audio_inputs = await this.feature_extractor!(audio, options);

            text = text.map((prompt: string) => prompt.replaceAll(this.audio_token, this.full_audio_sequence));
        }
        let image_inputs;
        if (images) {
            image_inputs = await this.image_processor!(images, options);
            text = text.map((prompt: string) => prompt.replaceAll(this.image_token, this.full_image_sequence));
        }

        let text_inputs = this.tokenizer!(text, options);
        return {
            ...text_inputs,
            ...image_inputs,
            ...audio_inputs,
        };
    }
}
