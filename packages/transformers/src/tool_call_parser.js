/**
 * @file Universal tool call parser for extracting structured tool/function calls
 * from LLM output text across different model architectures.
 *
 * Supports 20+ model families including Llama, Mistral, Qwen, Cohere, DeepSeek,
 * Hermes, Phi, Gemma, InternLM, Functionary, Gorilla, NexusRaven, xLAM, and more.
 *
 * @module tool_call_parser
 */

/**
 * @typedef {Object} ToolCall
 * @property {string} name The name of the function/tool being called.
 * @property {Object} arguments The arguments passed to the function.
 * @property {string|null} [id] An optional tool call ID (some models generate these).
 */

/**
 * @typedef {Object} ToolCallParserResult
 * @property {ToolCall[]} tool_calls The parsed tool calls.
 * @property {string} content Any non-tool-call text content from the response.
 * @property {string} parser The name of the parser that matched.
 */

// ============================================================================
// Individual parsers for each model family
// ============================================================================

/**
 * Llama 3.1/3.2/3.3 (Meta)
 *
 * Format: Uses `<|python_tag|>` followed by JSON, or generates JSON tool calls
 * directly after the assistant header. Supports both single and parallel tool calls.
 *
 * Single: `<|python_tag|>{"name": "func", "parameters": {"arg": "val"}}`
 * Parallel: `<|python_tag|>{"name": "func1", ...}\n{"name": "func2", ...}`
 * Function tag (3.2 lightweight): `<function=func>{"arg": "val"}</function>`
 * Also: `{"name": "func", "parameters": {...}}` without the python_tag
 */
function parseLlama(text) {
    const calls = [];
    let content = text;

    // Try <|python_tag|> format first
    const pythonTagMatch = text.match(/<\|python_tag\|>([\s\S]*?)(?:<\|eot_id\|>|<\|eom_id\|>|$)/);
    if (pythonTagMatch) {
        content = text.slice(0, text.indexOf('<|python_tag|>')).trim();
        const jsonSection = pythonTagMatch[1].trim();
        // Could be multiple JSON objects on separate lines
        for (const line of jsonSection.split('\n')) {
            const trimmed = line.trim();
            if (!trimmed) continue;
            try {
                const parsed = JSON.parse(trimmed);
                if (parsed.name) {
                    calls.push({
                        name: parsed.name,
                        arguments: parsed.parameters ?? parsed.arguments ?? {},
                        id: parsed.id ?? null,
                    });
                }
            } catch { /* skip non-JSON lines */ }
        }
    }

    // Try <function=name>{args}</function> format (Llama 3.2 lightweight 1B/3B)
    if (calls.length === 0) {
        const funcTagRegex = /<function=(\w+)>([\s\S]*?)<\/function>/g;
        let funcMatch;
        while ((funcMatch = funcTagRegex.exec(text)) !== null) {
            try {
                const args = JSON.parse(funcMatch[2].trim());
                calls.push({ name: funcMatch[1], arguments: args, id: null });
            } catch { /* skip malformed */ }
        }
        if (calls.length > 0) {
            content = text.replace(/<function=\w+>[\s\S]*?<\/function>/g, '').trim();
        }
    }

    // Also check for {"name":...} pattern at end of output (no python_tag)
    if (calls.length === 0) {
        const jsonPattern = /\{"name"\s*:\s*"[^"]+"\s*,\s*"parameters"\s*:\s*\{[\s\S]*?\}\s*\}/g;
        let match;
        while ((match = jsonPattern.exec(text)) !== null) {
            try {
                const parsed = JSON.parse(match[0]);
                calls.push({
                    name: parsed.name,
                    arguments: parsed.parameters ?? {},
                    id: parsed.id ?? null,
                });
            } catch { /* skip malformed */ }
        }
        if (calls.length > 0) {
            content = text.slice(0, text.indexOf(calls[0].name) - '{"name": "'.length).trim();
        }
    }

    return calls.length > 0 ? { tool_calls: calls, content, parser: 'llama' } : null;
}

/**
 * Mistral / Mixtral (Mistral AI)
 *
 * Format: `[TOOL_CALLS] [{"name": "func", "arguments": {...}, "id": "9charID"}]`
 * The [TOOL_CALLS] token marks the start. Tool call IDs are exactly 9 characters.
 */
function parseMistral(text) {
    const marker = '[TOOL_CALLS]';
    const idx = text.indexOf(marker);
    if (idx === -1) return null;

    const content = text.slice(0, idx).trim();
    const jsonStr = text.slice(idx + marker.length).trim();

    try {
        const parsed = JSON.parse(jsonStr);
        const arr = Array.isArray(parsed) ? parsed : [parsed];
        const calls = arr
            .filter((c) => c.name)
            .map((c) => ({
                name: c.name,
                arguments: c.arguments ?? c.parameters ?? {},
                id: c.id ?? null,
            }));
        if (calls.length > 0) {
            return { tool_calls: calls, content, parser: 'mistral' };
        }
    } catch { /* fall through */ }

    return null;
}

/**
 * Hermes (NousResearch) — also used by Qwen 2.5, Qwen 3, SOLAR, and others
 *
 * Format: `<tool_call>\n{"name": "func", "arguments": {...}}\n</tool_call>`
 * Tools are defined in `<tools>...</tools>` in the system prompt.
 * Multiple parallel calls use multiple `<tool_call>` blocks.
 */
function parseHermes(text) {
    const regex = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
    const calls = [];
    let match;
    while ((match = regex.exec(text)) !== null) {
        try {
            const parsed = JSON.parse(match[1].trim());
            calls.push({
                name: parsed.name,
                arguments: parsed.arguments ?? parsed.parameters ?? {},
                id: parsed.id ?? null,
            });
        } catch { /* skip malformed */ }
    }

    if (calls.length === 0) return null;

    const content = text.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();
    return { tool_calls: calls, content, parser: 'hermes' };
}

/**
 * Cohere Command-R / Command-R+
 *
 * Format: `Action: ```json\n[{"tool_name": "func", "parameters": {...}}]\n````
 * Also uses a simpler format: `Action: [{"tool_name": ..., "parameters": ...}]`
 */
function parseCohere(text) {
    // Try markdown code block format
    const blockMatch = text.match(/Action:\s*```(?:json)?\s*\n?([\s\S]*?)\n?\s*```/);
    // Try inline format
    const inlineMatch = text.match(/Action:\s*(\[[\s\S]*?\])\s*$/m);

    const jsonStr = blockMatch?.[1] ?? inlineMatch?.[1];
    if (!jsonStr) return null;

    try {
        const parsed = JSON.parse(jsonStr.trim());
        const arr = Array.isArray(parsed) ? parsed : [parsed];
        const calls = arr
            .filter((c) => c.tool_name || c.name)
            .map((c) => ({
                name: c.tool_name ?? c.name,
                arguments: c.parameters ?? c.arguments ?? {},
                id: c.id ?? null,
            }));
        if (calls.length > 0) {
            const actionIdx = text.indexOf('Action:');
            const content = text.slice(0, actionIdx).trim();
            return { tool_calls: calls, content, parser: 'cohere' };
        }
    } catch { /* fall through */ }

    return null;
}

/**
 * DeepSeek V2/V3/V3.1
 *
 * V2 format: `<｜tool▁call▁begin｜>function_name\n```json\n{...}\n```<｜tool▁call▁end｜>`
 * V3.1 format: `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>name<｜tool▁sep｜>{args}<｜tool▁call▁end｜><｜tool▁calls▁end｜>`
 * Uses fullwidth characters in special tokens. Supports parallel calls.
 */
function parseDeepSeek(text) {
    const calls = [];

    // Helper to match both fullwidth ｜ and ASCII | bar variants, and ▁ or space
    const bar = '(?:｜|\\|)';
    const sep = '[\\s\u2581]';

    // Try V3.1 format first: name<｜tool▁sep｜>{args}
    const v31Regex = new RegExp(
        `<${bar}tool${sep}call${sep}begin${bar}>\\s*(\\w+)\\s*<${bar}tool${sep}sep${bar}>\\s*([\\s\\S]*?)\\s*<${bar}tool${sep}call${sep}end${bar}>`,
        'g',
    );
    let match;
    while ((match = v31Regex.exec(text)) !== null) {
        try {
            const args = JSON.parse(match[2].trim());
            calls.push({ name: match[1], arguments: args, id: null });
        } catch { /* skip malformed */ }
    }

    // Try V2 format: name\n```json\n{args}\n```
    if (calls.length === 0) {
        const v2Regex = new RegExp(
            `<${bar}tool${sep}call${sep}begin${bar}>\\s*(\\w+)\\s*\\n\`\`\`(?:json)?\\n([\\s\\S]*?)\\n\`\`\`\\s*<${bar}tool${sep}call${sep}end${bar}>`,
            'g',
        );
        while ((match = v2Regex.exec(text)) !== null) {
            try {
                const args = JSON.parse(match[2].trim());
                calls.push({ name: match[1], arguments: args, id: null });
            } catch { /* skip malformed */ }
        }
    }

    if (calls.length === 0) return null;

    // Strip all tool call blocks (both outer plural and inner singular markers)
    const content = text
        .replace(new RegExp(`<${bar}tool${sep}calls?${sep}(?:begin|end)${bar}>`, 'g'), '')
        .replace(new RegExp(`<${bar}tool${sep}call${sep}(?:begin|end)${bar}>[\\s\\S]*?<${bar}tool${sep}call${sep}end${bar}>`, 'g'), '')
        .replace(new RegExp(`<${bar}tool${sep}sep${bar}>`, 'g'), '')
        .trim();
    return { tool_calls: calls, content, parser: 'deepseek' };
}

/**
 * Phi-4 / Phi-4-mini (Microsoft)
 *
 * Format: `<|tool_calls|>[{"name": "func", "arguments": {...}}]<|/tool_calls|>`
 * Supports parallel calls via JSON array. Phi-3 has no native tool calling.
 */
function parsePhi(text) {
    const match = text.match(/<\|tool_calls\|>\s*([\s\S]*?)\s*<\|\/tool_calls\|>/);
    if (!match) return null;

    try {
        const parsed = JSON.parse(match[1].trim());
        const arr = Array.isArray(parsed) ? parsed : [parsed];
        const calls = arr
            .filter((c) => c.name)
            .map((c) => ({
                name: c.name,
                arguments: c.arguments ?? c.parameters ?? {},
                id: c.id ?? null,
            }));
        if (calls.length > 0) {
            const content = text.slice(0, text.indexOf('<|tool_calls|>')).trim();
            return { tool_calls: calls, content, parser: 'phi' };
        }
    } catch { /* fall through */ }

    return null;
}

/**
 * Phi-3 functools format (legacy)
 *
 * Format: `functools[{"name": "func", "arguments": {...}}]`
 */
function parsePhiFunctools(text) {
    const match = text.match(/functools\s*(\[[\s\S]*?\])/);
    if (!match) return null;

    try {
        const parsed = JSON.parse(match[1].trim());
        const arr = Array.isArray(parsed) ? parsed : [parsed];
        const calls = arr
            .filter((c) => c.name)
            .map((c) => ({
                name: c.name,
                arguments: c.arguments ?? c.parameters ?? {},
                id: c.id ?? null,
            }));
        if (calls.length > 0) {
            const content = text.slice(0, text.indexOf('functools')).trim();
            return { tool_calls: calls, content, parser: 'phi_functools' };
        }
    } catch { /* fall through */ }

    return null;
}

/**
 * InternLM 2 / 2.5 (Shanghai AI Lab)
 *
 * Format: `<|action_start|><|plugin|>\n{"name": "func", "parameters": {...}}<|action_end|>`
 * Uses special tokens: <|action_start|>, <|action_end|>, <|plugin|>, <|interpreter|>
 */
function parseInternLM(text) {
    const regex = /<\|action_start\|>\s*<\|plugin\|>\s*([\s\S]*?)\s*<\|action_end\|>/g;
    const calls = [];
    let match;
    while ((match = regex.exec(text)) !== null) {
        try {
            const parsed = JSON.parse(match[1].trim());
            calls.push({
                name: parsed.name,
                arguments: parsed.parameters ?? parsed.arguments ?? {},
                id: parsed.id ?? null,
            });
        } catch { /* skip malformed */ }
    }

    if (calls.length === 0) return null;

    const content = text
        .replace(/<\|action_start\|>\s*<\|plugin\|>[\s\S]*?<\|action_end\|>/g, '')
        .trim();
    return { tool_calls: calls, content, parser: 'internlm' };
}

/**
 * ChatGLM / GLM-4 (Zhipu AI)
 *
 * Format: The model outputs the function name followed by newline and JSON arguments.
 * `func_name\n{"arg": "val"}`
 * In the chat template, tool calls are wrapped with the `observation` role.
 */
function parseChatGLM(text) {
    // GLM-4 uses a specific pattern where the assistant outputs function name + args
    const match = text.match(/^(\w+)\n(\{[\s\S]*\})\s*$/m);
    if (!match) return null;

    try {
        const args = JSON.parse(match[2].trim());
        return {
            tool_calls: [{ name: match[1], arguments: args, id: null }],
            content: '',
            parser: 'chatglm',
        };
    } catch {
        return null;
    }
}

/**
 * Functionary (MeetKai)
 *
 * Format: Uses `>>>` prefix with function name, then JSON arguments.
 * `>>>func_name\n{"arg": "val"}`
 * Also uses `all` as a special function name for regular text.
 */
function parseFunctionary(text) {
    const regex = />>>\s*(\w+)\s*\n([\s\S]*?)(?=>>>|$)/g;
    const calls = [];
    let content = '';
    let match;

    while ((match = regex.exec(text)) !== null) {
        const funcName = match[1].trim();
        const body = match[2].trim();

        if (funcName === 'all') {
            content += body;
            continue;
        }

        try {
            const args = JSON.parse(body);
            calls.push({ name: funcName, arguments: args, id: null });
        } catch {
            // If not JSON, might be text content for this function
            calls.push({ name: funcName, arguments: { content: body }, id: null });
        }
    }

    if (calls.length === 0) return null;
    return { tool_calls: calls, content: content.trim(), parser: 'functionary' };
}

/**
 * Gorilla (Berkeley)
 *
 * Format: `<<function>>func_name(arg1="val1", arg2=val2)`
 * Arguments can be string-quoted or bare values.
 */
function parseGorilla(text) {
    const regex = /<<function>>\s*(\w+)\(([^)]*)\)/g;
    const calls = [];
    let match;

    while ((match = regex.exec(text)) !== null) {
        const name = match[1];
        const argsStr = match[2].trim();
        const args = {};

        if (argsStr) {
            // Parse key=value pairs
            const argRegex = /(\w+)\s*=\s*(?:"([^"]*?)"|'([^']*?)'|(\S+?))\s*(?:,|$)/g;
            let argMatch;
            while ((argMatch = argRegex.exec(argsStr)) !== null) {
                const key = argMatch[1];
                const value = argMatch[2] ?? argMatch[3] ?? argMatch[4];
                // Try to parse as number/boolean
                if (value === 'true') args[key] = true;
                else if (value === 'false') args[key] = false;
                else if (!isNaN(Number(value)) && value !== '') args[key] = Number(value);
                else args[key] = value;
            }
        }

        calls.push({ name, arguments: args, id: null });
    }

    if (calls.length === 0) return null;

    const content = text.replace(/<<function>>\s*\w+\([^)]*\)/g, '').trim();
    return { tool_calls: calls, content, parser: 'gorilla' };
}

/**
 * NexusRaven (Nexusflow)
 *
 * Format: `Call: func_name(arg1="val1", arg2=val2)\nThought: reasoning...`
 */
function parseNexusRaven(text) {
    const regex = /Call:\s*(\w+)\(([^)]*)\)/g;
    const calls = [];
    let match;

    while ((match = regex.exec(text)) !== null) {
        const name = match[1];
        const argsStr = match[2].trim();
        const args = {};

        if (argsStr) {
            const argRegex = /(\w+)\s*=\s*(?:"([^"]*?)"|'([^']*?)'|(\S+?))\s*(?:,|$)/g;
            let argMatch;
            while ((argMatch = argRegex.exec(argsStr)) !== null) {
                const key = argMatch[1];
                const value = argMatch[2] ?? argMatch[3] ?? argMatch[4];
                if (value === 'true') args[key] = true;
                else if (value === 'false') args[key] = false;
                else if (!isNaN(Number(value)) && value !== '') args[key] = Number(value);
                else args[key] = value;
            }
        }

        calls.push({ name, arguments: args, id: null });
    }

    if (calls.length === 0) return null;

    // Extract thought/content
    const thoughtMatch = text.match(/Thought:\s*([\s\S]*?)(?:Call:|$)/);
    const content = thoughtMatch?.[1]?.trim() ?? text.replace(/Call:\s*\w+\([^)]*\)/g, '').trim();
    return { tool_calls: calls, content, parser: 'nexusraven' };
}

/**
 * xLAM (Salesforce)
 *
 * Format: Raw JSON array of tool calls: `[{"name": "func", "arguments": {...}}]`
 * May be wrapped in ```json code blocks.
 */
function parseXLAM(text) {
    // Try code block wrapped JSON first
    const codeBlockMatch = text.match(/```(?:json)?\s*\n?(\[[\s\S]*?\])\n?\s*```/);
    const jsonStr = codeBlockMatch?.[1] ?? text.trim();

    // Only try parsing if it looks like a JSON array
    if (!jsonStr.trimStart().startsWith('[')) return null;

    try {
        const parsed = JSON.parse(jsonStr);
        if (!Array.isArray(parsed)) return null;

        const calls = parsed
            .filter((c) => c.name)
            .map((c) => ({
                name: c.name,
                arguments: c.arguments ?? c.parameters ?? {},
                id: c.id ?? null,
            }));

        if (calls.length > 0) {
            const content = codeBlockMatch
                ? text.slice(0, text.indexOf('```')).trim()
                : '';
            return { tool_calls: calls, content, parser: 'xlam' };
        }
    } catch { /* fall through */ }

    return null;
}

/**
 * FireFunction (Fireworks AI)
 *
 * Format: Similar to OpenAI — JSON with `functools` wrapper or direct JSON array.
 * `functools[{"name": "func", "arguments": {...}}]`
 * Also supports: `{"tool_calls": [{"function": {"name": "...", "arguments": "..."}}]}`
 */
function parseFireFunction(text) {
    // Try OpenAI-style function call format
    const openaiMatch = text.match(/\{"tool_calls"\s*:\s*(\[[\s\S]*?\])\s*\}/);
    if (openaiMatch) {
        try {
            const parsed = JSON.parse(openaiMatch[1]);
            const calls = parsed
                .filter((c) => c.function?.name)
                .map((c) => ({
                    name: c.function.name,
                    arguments: typeof c.function.arguments === 'string'
                        ? JSON.parse(c.function.arguments)
                        : c.function.arguments ?? {},
                    id: c.id ?? null,
                }));
            if (calls.length > 0) {
                return { tool_calls: calls, content: '', parser: 'firefunction' };
            }
        } catch { /* fall through */ }
    }

    return null;
}

/**
 * Granite (IBM)
 *
 * Format: Uses Hermes-style `<tool_call>` tags, or `<|tool_call|>` special tokens.
 * Fallback: `{"name": "func", "arguments": {...}}`
 */
function parseGranite(text) {
    // Try <|tool_call|> format (IBM-specific markers)
    const regex = /<\|tool_call\|>\s*([\s\S]*?)\s*(?:<\|\/tool_call\|>|<\|end_of_text\|>|$)/g;
    const calls = [];
    let match;
    while ((match = regex.exec(text)) !== null) {
        try {
            const parsed = JSON.parse(match[1].trim());
            calls.push({
                name: parsed.name,
                arguments: parsed.arguments ?? parsed.parameters ?? {},
                id: parsed.id ?? null,
            });
        } catch { /* skip malformed */ }
    }

    if (calls.length === 0) return null;

    const content = text.replace(/<\|tool_call\|>[\s\S]*?(?:<\|\/tool_call\|>|$)/g, '').trim();
    return { tool_calls: calls, content, parser: 'granite' };
}

/**
 * Gemma 2/3 (Google) — prompt-based, no dedicated tokens
 *
 * Format: Relies on prompt instructions. Common formats:
 * - JSON: `{"name": "func", "parameters": {...}}`
 * - tool_code block: ```tool_code\nfunc(arg=val)\n```
 */
function parseGemma(text) {
    // Try tool_code block format
    const codeMatch = text.match(/```tool_code\s*\n([\s\S]*?)\n\s*```/);
    if (codeMatch) {
        const code = codeMatch[1].trim();
        // Parse function call syntax: func(arg1=val1, arg2=val2)
        const funcMatch = code.match(/^(\w+)\(([\s\S]*)\)$/);
        if (funcMatch) {
            const name = funcMatch[1];
            const argsStr = funcMatch[2].trim();
            const args = {};
            if (argsStr) {
                const argRegex = /(\w+)\s*=\s*(?:"([^"]*?)"|'([^']*?)'|(\S+?))\s*(?:,|$)/g;
                let argMatch;
                while ((argMatch = argRegex.exec(argsStr)) !== null) {
                    const key = argMatch[1];
                    const value = argMatch[2] ?? argMatch[3] ?? argMatch[4];
                    if (value === 'true') args[key] = true;
                    else if (value === 'false') args[key] = false;
                    else if (!isNaN(Number(value)) && value !== '') args[key] = Number(value);
                    else args[key] = value;
                }
            }
            const content = text.replace(/```tool_code[\s\S]*?```/g, '').trim();
            return { tool_calls: [{ name, arguments: args, id: null }], content, parser: 'gemma' };
        }
    }

    return null;
}

/**
 * FunctionGemma (Google, specialized 270M model)
 *
 * Format: `<start_function_call>call:func_name{key:<escape>value<escape>}<end_function_call>`
 */
function parseFunctionGemma(text) {
    const regex = /<start_function_call>\s*call:(\w+)\{([\s\S]*?)\}\s*<end_function_call>/g;
    const calls = [];
    let match;

    while ((match = regex.exec(text)) !== null) {
        const name = match[1];
        const argsStr = match[2];
        const args = {};

        // Parse key:<escape>value<escape> pairs
        const argRegex = /(\w+):<escape>([\s\S]*?)<escape>/g;
        let argMatch;
        while ((argMatch = argRegex.exec(argsStr)) !== null) {
            const value = argMatch[2];
            if (value === 'true') args[argMatch[1]] = true;
            else if (value === 'false') args[argMatch[1]] = false;
            else if (!isNaN(Number(value)) && value !== '') args[argMatch[1]] = Number(value);
            else args[argMatch[1]] = value;
        }

        calls.push({ name, arguments: args, id: null });
    }

    if (calls.length === 0) return null;

    const content = text
        .replace(/<start_function_call>[\s\S]*?<end_function_call>/g, '')
        .trim();
    return { tool_calls: calls, content, parser: 'functiongemma' };
}

/**
 * Jamba (AI21)
 *
 * Format: `<tool_calls>[{"name": "func", "arguments": {...}}]</tool_calls>`
 * Also supports OpenAI-compatible: `{"tool_calls": [{"function": {"name": "...", "arguments": "..."}}]}`
 */
function parseJamba(text) {
    // Try <tool_calls> tag format first
    const tagMatch = text.match(/<tool_calls>\s*([\s\S]*?)\s*<\/tool_calls>/);
    if (tagMatch) {
        try {
            const parsed = JSON.parse(tagMatch[1].trim());
            const arr = Array.isArray(parsed) ? parsed : [parsed];
            const calls = arr
                .filter((c) => c.name)
                .map((c) => ({
                    name: c.name,
                    arguments: typeof c.arguments === 'string'
                        ? JSON.parse(c.arguments)
                        : c.arguments ?? c.parameters ?? {},
                    id: c.id ?? null,
                }));
            if (calls.length > 0) {
                const content = text.slice(0, text.indexOf('<tool_calls>')).trim();
                return { tool_calls: calls, content, parser: 'jamba' };
            }
        } catch { /* fall through */ }
    }

    // Fall back to OpenAI-compatible format
    return parseFireFunction(text);
}

// ============================================================================
// Model family detection
// ============================================================================

/**
 * Map of model family identifiers to their ordered list of parsers.
 * Each entry maps a detection key to an array of parser functions to try in order.
 * @type {Record<string, Array<function(string): ToolCallParserResult|null>>}
 */
const MODEL_PARSERS = {
    llama: [parseLlama, parseHermes],
    mistral: [parseMistral, parseHermes],
    mixtral: [parseMistral, parseHermes],
    qwen: [parseHermes, parseLlama],
    qwen2: [parseHermes, parseLlama],
    qwen3: [parseHermes, parseLlama],
    cohere: [parseCohere, parseHermes],
    command: [parseCohere, parseHermes],
    deepseek: [parseDeepSeek, parseHermes],
    hermes: [parseHermes],
    phi: [parsePhi, parsePhiFunctools, parseHermes],
    internlm: [parseInternLM, parseHermes],
    chatglm: [parseChatGLM],
    glm: [parseChatGLM],
    gemma: [parseFunctionGemma, parseGemma, parseHermes],
    functionary: [parseFunctionary],
    gorilla: [parseGorilla],
    nexusraven: [parseNexusRaven],
    xlam: [parseXLAM],
    firefunction: [parseFireFunction, parsePhiFunctools],
    granite: [parseGranite, parseHermes],
    solar: [parseHermes],
    jamba: [parseJamba, parseHermes],
    yi: [parseHermes, parseLlama],
    falcon: [parseHermes, parseLlama],
};

/**
 * Default parser chain used when the model family cannot be determined.
 * Tries all major formats in order of specificity (most distinctive markers first).
 */
const DEFAULT_PARSER_CHAIN = [
    parsePhi,
    parseMistral,
    parseDeepSeek,
    parseInternLM,
    parseGranite,
    parseFunctionGemma,
    parseHermes,
    parseCohere,
    parseFunctionary,
    parseGorilla,
    parseNexusRaven,
    parseFireFunction,
    parsePhiFunctools,
    parseLlama,
    parseGemma,
    parseXLAM,
];

/**
 * Detect the model family from a tokenizer instance or model name string.
 *
 * @param {import('./tokenization_utils.js').PreTrainedTokenizer|string} tokenizerOrName
 *   A tokenizer instance (checks `name_or_path` and config) or a model name string.
 * @returns {string|null} The detected model family key, or null if unknown.
 */
function detectModelFamily(tokenizerOrName) {
    let name = '';

    if (typeof tokenizerOrName === 'string') {
        name = tokenizerOrName.toLowerCase();
    } else if (tokenizerOrName) {
        // Try to extract model name from tokenizer config
        const config = tokenizerOrName.config ?? {};
        name = (
            config.name_or_path ??
            config._name_or_path ??
            config.model_type ??
            tokenizerOrName.name_or_path ??
            ''
        ).toLowerCase();
    }

    if (!name) return null;

    for (const family of Object.keys(MODEL_PARSERS)) {
        if (name.includes(family)) {
            return family;
        }
    }

    return null;
}

// ============================================================================
// Public API
// ============================================================================

/**
 * Parse tool calls from LLM output text.
 *
 * Automatically detects the model family from the tokenizer and applies the
 * appropriate parser(s). Falls back to trying all known formats if the model
 * family cannot be determined.
 *
 * @param {string} text The raw decoded text output from the model.
 * @param {Object} [options] Options for parsing.
 * @param {import('./tokenization_utils.js').PreTrainedTokenizer|null} [options.tokenizer=null]
 *   The tokenizer instance, used to auto-detect the model family.
 * @param {string|null} [options.model=null]
 *   The model name/ID string, used to auto-detect the model family if no tokenizer is provided.
 * @param {string|null} [options.parser=null]
 *   Force a specific parser by name (e.g., 'hermes', 'mistral', 'llama').
 *   Overrides auto-detection.
 * @returns {ToolCallParserResult} The parsed result with tool_calls array, remaining content, and parser name.
 *
 * @example
 * // Auto-detect from tokenizer
 * import { AutoTokenizer } from '@huggingface/transformers';
 * import { parseToolCalls } from '@huggingface/transformers';
 *
 * const tokenizer = await AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct');
 * const result = parseToolCalls(modelOutput, { tokenizer });
 * for (const call of result.tool_calls) {
 *   console.log(call.name, call.arguments);
 * }
 *
 * @example
 * // Force a specific parser
 * const result = parseToolCalls(modelOutput, { parser: 'hermes' });
 *
 * @example
 * // Auto-detect from model name
 * const result = parseToolCalls(modelOutput, { model: 'Qwen/Qwen2.5-7B-Instruct' });
 */
export function parseToolCalls(text, { tokenizer = null, model = null, parser = null } = {}) {
    if (!text || typeof text !== 'string') {
        return { tool_calls: [], content: text ?? '', parser: 'none' };
    }

    // Determine which parsers to try
    let parsersToTry;

    if (parser) {
        // User explicitly requested a parser
        const key = parser.toLowerCase();
        parsersToTry = MODEL_PARSERS[key];
        if (!parsersToTry) {
            throw new Error(
                `Unknown parser "${parser}". Available parsers: ${Object.keys(MODEL_PARSERS).join(', ')}`
            );
        }
    } else {
        // Auto-detect from tokenizer or model name
        const family = detectModelFamily(tokenizer ?? model);
        parsersToTry = family ? MODEL_PARSERS[family] : DEFAULT_PARSER_CHAIN;
    }

    // Try each parser in order
    for (const parserFn of parsersToTry) {
        const result = parserFn(text);
        if (result) return result;
    }

    // No tool calls found
    return { tool_calls: [], content: text, parser: 'none' };
}

/**
 * Check if text contains tool calls without fully parsing them.
 * Faster than `parseToolCalls` when you only need to know if tool calls exist.
 *
 * @param {string} text The raw decoded text output from the model.
 * @returns {boolean} True if the text likely contains tool calls.
 */
export function hasToolCalls(text) {
    if (!text) return false;
    return (
        text.includes('<tool_call>') ||
        text.includes('[TOOL_CALLS]') ||
        text.includes('<|python_tag|>') ||
        text.includes('<function=') ||
        text.includes('<|tool_calls|>') ||
        text.includes('<tool_calls>') ||
        text.includes('<|action_start|>') ||
        text.includes('<<function>>') ||
        text.includes('>>>') ||
        text.includes('Call:') ||
        text.includes('Action:') ||
        text.includes('functools') ||
        text.includes('<start_function_call>') ||
        text.includes('<|tool_call|>') ||
        /tool[\s\u2581]call[\s\u2581]begin/.test(text)
    );
}

/**
 * Get the list of available parser names.
 *
 * @returns {string[]} Array of parser name strings that can be passed to `parseToolCalls`.
 */
export function getAvailableParsers() {
    return Object.keys(MODEL_PARSERS);
}
