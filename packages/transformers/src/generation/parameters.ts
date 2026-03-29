/**
 * @module generation/parameters
 */

import type { Tensor } from '../utils/tensor';
import type { GenerationConfig } from './configuration_utils';
import type { LogitsProcessorList } from './logits_process';
import type { StoppingCriteria, StoppingCriteriaList } from './stopping_criteria';
import type { BaseStreamer } from './streamers';

export interface GenerationFunctionParametersBase {
    inputs?: Tensor | null;
    generation_config?: GenerationConfig | null;
    logits_processor?: LogitsProcessorList | null;
    stopping_criteria?: StoppingCriteria | StoppingCriteria[] | StoppingCriteriaList | null;
    streamer?: BaseStreamer | null;
    decoder_input_ids?: number[] | null;
}

export type GenerationFunctionParameters = GenerationFunctionParametersBase & Partial<GenerationConfig> & Record<string, any>;
