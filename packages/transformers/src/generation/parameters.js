/**
 * @module generation/parameters
 */

/**
 * @typedef {import('../utils/tensor.js').Tensor} Tensor
 * @typedef {import('./configuration_utils.js').GenerationConfig} GenerationConfig
 * @typedef {import('./logits_process.js').LogitsProcessorList} LogitsProcessorList
 * @typedef {import('./stopping_criteria.js').StoppingCriteria} StoppingCriteria
 * @typedef {import('./stopping_criteria.js').StoppingCriteriaList} StoppingCriteriaList
 * @typedef {import('./streamers.js').BaseStreamer} BaseStreamer
 */

/**
 * @typedef {Object} GenerationFunctionParametersBase
 * @property {Tensor} [inputs]
 * @property {GenerationConfig} [generation_config]
 * @property {LogitsProcessorList} [logits_processor]
 * @property {StoppingCriteria|StoppingCriteria[]|StoppingCriteriaList} [stopping_criteria]
 * @property {BaseStreamer} [streamer]
 * @property {number[]} [decoder_input_ids]
 */

/**
 * @typedef {GenerationFunctionParametersBase & Partial<GenerationConfig> & Record<string, any>} GenerationFunctionParameters
 */

// Ensure this file is treated as a module
export {};
