type Fetcher = typeof fetch;

export type ChatKitProviderBindings = {
  CHATKIT_UPSTREAM_URL?: string;
  CHATKIT_ORIGIN?: string;
  CHATKIT_REFERER?: string;
  PROVIDER_API_KEY?: string;
  DEFAULT_MODEL?: string;
  MODEL_LIST?: string;
};

type ProviderOptions = {
  fetch?: Fetcher;
};

type ChatCompletionRequest = {
  model?: string;
  messages?: ChatMessage[];
  stream?: boolean;
  tools?: ChatTool[];
  tool_choice?: "auto" | "none" | "required" | Record<string, unknown>;
};

type ResponsesRequest = {
  model?: string;
  instructions?: unknown;
  input?: unknown;
  stream?: boolean;
  tools?: ChatTool[];
  tool_choice?: "auto" | "none" | "required" | Record<string, unknown>;
};

type ChatMessage = {
  role?: string;
  type?: string;
  content?: unknown;
  id?: string;
  call_id?: string;
  name?: string;
  tool_call_id?: string;
  tool_calls?: unknown;
};

type ChatTool = {
  type?: string;
  name?: string;
  description?: string;
  parameters?: unknown;
  format?: {
    type?: string;
    syntax?: string;
    definition?: string;
  };
  tools?: ChatTool[];
  function?: {
    name?: string;
    description?: string;
    parameters?: unknown;
  };
};

type PromptTool = PromptFunctionTool | PromptCustomTool;

type PromptFunctionTool = {
  type: "function";
  name: string;
  description: string;
  parameters: unknown;
  namespace?: string;
};

type PromptCustomTool = {
  type: "custom";
  name: string;
  description: string;
  format?: {
    type?: string;
    syntax?: string;
    definition?: string;
  };
};

type ChatKitInput = {
  content: Array<{ type: "input_text"; text: string }>;
  quoted_text: "";
  attachments: [];
  inference_options: {
    model: string;
  };
};

type AssistantDecision =
  | {
      type: "final";
      content: string;
    }
  | {
      type: "tool_calls";
      tool_calls: ToolCallIntent[];
    };

type ToolCallIntent = {
  id?: string;
  name: string;
  namespace?: string;
  arguments?: unknown;
  input?: string;
  kind?: "function" | "custom";
};

type AssistantDecisionResolution =
  | {
      ok: true;
      decision: AssistantDecision;
    }
  | {
      ok: false;
      message: string;
    };

type TaggedParseResult =
  | {
      ok: true;
      decision: AssistantDecision;
    }
  | {
      ok: false;
      message: string;
    };

type NormalizedToolChoice =
  | {
      type: "auto" | "required" | "none";
    }
  | {
      type: "function";
      name: string;
    };

type ChatKitReadResult = {
  text: string;
  threadId?: string;
};

type ChatKitDecisionResult =
  | {
      ok: true;
      decision: AssistantDecision;
    }
  | {
      ok: false;
      response: Response;
    };

const DEFAULT_CHATKIT_UPSTREAM_URL = "https://chatkit-studio-internal.onrender.com/chatkit";
const DEFAULT_CHATKIT_ORIGIN = "https://chatkit.studio";
const DEFAULT_MODEL = "gpt-5";
const DEFAULT_MODELS = ["gpt-5", "gpt-5-nano", "gpt-5-reason", "gpt-5-pro"];
const MAX_CHATKIT_TEXT_BYTES = 512 * 1024;
const MAX_CHATKIT_SSE_RECORD_BYTES = 2 * 1024 * 1024;
const MAX_PROTOCOL_REPAIR_PREVIEW_CHARS = 4000;

export function createChatKitProvider(options: ProviderOptions = {}) {
  const upstreamFetch = options.fetch ?? fetch;

  return {
    async fetch(request: Request, env: ChatKitProviderBindings = {}): Promise<Response> {
      try {
        if (request.method === "OPTIONS") {
          return cors(new Response(null, { status: 204 }));
        }

        if (!(await isAuthorized(request, env))) {
          return jsonError("invalid_api_key", "Invalid API key.", 401);
        }

        const url = new URL(request.url);

        if (request.method === "GET" && url.pathname === "/health") {
          return json({ ok: true });
        }

        if (request.method === "GET" && url.pathname === "/v1/models") {
          const models = supportedModels(env);
          return json({
            object: "list",
            data: models.map((id) => ({ id, object: "model", owned_by: "chatkit" })),
            models: models.map(codexModelInfo)
          });
        }

        if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
          const body = (await request.json()) as ChatCompletionRequest;
          return handleChatCompletions(body, env, upstreamFetch);
        }

        if (request.method === "POST" && url.pathname === "/v1/responses") {
          const body = (await request.json()) as ResponsesRequest;
          return handleResponses(body, env, upstreamFetch);
        }

        return jsonError("not_found", "Not found.", 404);
      } catch (error) {
        console.error(JSON.stringify({ level: "error", message: String(error) }));
        return jsonError("internal_error", "Internal server error.", 500);
      }
    }
  };
}

export function buildChatKitInput(request: ChatCompletionRequest | ResponsesRequest): ChatKitInput {
  const model = nonEmptyString(request.model) || DEFAULT_MODEL;
  const isChatCompletion = isChatCompletionRequest(request);
  const messages = isChatCompletion ? request.messages || [] : responsesInputToMessages(request.input);
  const instructions = isChatCompletion ? undefined : request.instructions;
  const text = buildPrompt(messages, request.tools || [], request.tool_choice, instructions);

  return {
    content: [{ type: "input_text", text }],
    quoted_text: "",
    attachments: [],
    inference_options: { model }
  };
}

function isChatCompletionRequest(
  request: ChatCompletionRequest | ResponsesRequest
): request is ChatCompletionRequest {
  return "messages" in request;
}

export function parseAssistantDecision(text: string): AssistantDecision {
  const tagged = parseTaggedAssistantDecision(text);
  if (tagged.ok) {
    return tagged.decision;
  }

  const trimmed = unwrapMarkdownFence(text.trim());
  const parsed = parseJsonObject(trimmed);

  if (isRecord(parsed) && parsed.type === "tool_calls" && Array.isArray(parsed.tool_calls)) {
    const calls = parsed.tool_calls
      .map((call) => normalizeToolCall(call))
      .filter((call): call is ToolCallIntent => call !== undefined);

    if (calls.length > 0) {
      return { type: "tool_calls", tool_calls: calls };
    }
  }

  if (isRecord(parsed) && parsed.type === "final" && typeof parsed.content === "string") {
    return { type: "final", content: parsed.content };
  }

  return { type: "final", content: text.trim() };
}

function resolveAssistantDecision(
  text: string,
  tools: ChatTool[] = [],
  toolChoice: unknown
): AssistantDecisionResolution {
  const normalizedTools = normalizeToolsForPrompt(tools);
  const normalizedChoice = normalizeToolChoice(toolChoice);
  const toolsAreActive = normalizedTools.length > 0 && normalizedChoice.type !== "none";
  const tagged = parseTaggedAssistantDecision(text);

  if (toolsAreActive && !tagged.ok) {
    return {
      ok: false,
      message:
        "ChatKit assistant did not follow the tagged tool protocol. Expected exactly one <tool_calls> or <final_answer> block."
    };
  }

  const decision = tagged.ok ? tagged.decision : parseAssistantDecision(text);
  attachToolKinds(decision, normalizedTools);
  const validationError = validateDecisionAgainstTools(decision, normalizedTools, normalizedChoice);
  if (validationError) {
    return { ok: false, message: validationError };
  }

  return { ok: true, decision };
}

function attachToolKinds(decision: AssistantDecision, tools: PromptTool[]): void {
  if (decision.type !== "tool_calls") {
    return;
  }

  for (const call of decision.tool_calls) {
    const tool = findPromptToolForCall(tools, call);
    if (tool) {
      call.kind = tool.type;
      if (tool.type === "custom" && call.input === undefined) {
        call.input = customToolArgumentsToInput(call.arguments);
      }
    } else if (call.input !== undefined) {
      call.kind = "custom";
    } else {
      call.kind = "function";
    }
  }
}

function customToolArgumentsToInput(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  if (isRecord(value)) {
    if (typeof value.input === "string") {
      return value.input;
    }
    if (typeof value.code === "string") {
      return value.code;
    }
    if (typeof value.command === "string") {
      return value.command;
    }
    if (typeof value.cmd === "string") {
      return value.cmd;
    }
  }

  if (value === undefined) {
    return "";
  }

  return JSON.stringify(value);
}

function findPromptToolForCall(tools: PromptTool[], call: ToolCallIntent): PromptTool | undefined {
  return tools.find((tool) => tool.name === call.name && ("namespace" in tool ? tool.namespace === call.namespace : true));
}

function parseTaggedAssistantDecision(text: string): TaggedParseResult {
  const normalized = stripLeadingThinkBlocks(unwrapMarkdownFence(text.trim())).trim();

  const toolCallsText = exactTaggedContent(normalized, "tool_calls");
  if (toolCallsText !== undefined) {
    const parsed = parseJsonArray(toolCallsText.trim());
    if (!Array.isArray(parsed)) {
      return { ok: false, message: "<tool_calls> must contain a JSON array." };
    }

    const calls = parsed
      .map((call) => normalizeToolCall(call))
      .filter((call): call is ToolCallIntent => call !== undefined);

    if (calls.length === 0) {
      return { ok: false, message: "<tool_calls> did not contain any valid tool call." };
    }

    return { ok: true, decision: { type: "tool_calls", tool_calls: calls } };
  }

  const finalText = exactTaggedContent(normalized, "final_answer");
  if (finalText !== undefined) {
    return { ok: true, decision: { type: "final", content: finalText } };
  }

  return { ok: false, message: "No tagged assistant decision found." };
}

function stripLeadingThinkBlocks(text: string): string {
  let remaining = text.trim();

  while (remaining.startsWith("<think>")) {
    const end = remaining.indexOf("</think>");
    if (end < 0) {
      return remaining;
    }
    remaining = remaining.slice(end + "</think>".length).trim();
  }

  return remaining;
}

function exactTaggedContent(text: string, tag: "tool_calls" | "final_answer"): string | undefined {
  const open = `<${tag}>`;
  const close = `</${tag}>`;

  if (!text.startsWith(open) || !text.endsWith(close)) {
    return undefined;
  }

  return text.slice(open.length, text.length - close.length);
}

async function handleChatCompletions(
  body: ChatCompletionRequest,
  env: ChatKitProviderBindings,
  upstreamFetch: Fetcher
): Promise<Response> {
  const model = nonEmptyString(body.model) || env.DEFAULT_MODEL || DEFAULT_MODEL;
  const modelError = validateModel(model, env);
  if (modelError) {
    return modelError;
  }
  const input = buildChatKitInput({ ...body, model });
  const resolved = await resolveDecisionFromChatKit(input, body.tools, body.tool_choice, env, upstreamFetch);
  if (!resolved.ok) {
    return resolved.response;
  }

  const completion = chatCompletionFromDecision(resolved.decision, model);

  if (body.stream) {
    return chatCompletionStream(completion);
  }

  return json(completion);
}

async function handleResponses(
  body: ResponsesRequest,
  env: ChatKitProviderBindings,
  upstreamFetch: Fetcher
): Promise<Response> {
  const model = nonEmptyString(body.model) || env.DEFAULT_MODEL || DEFAULT_MODEL;
  const modelError = validateModel(model, env);
  if (modelError) {
    return modelError;
  }
  const input = buildChatKitInput({ ...body, model });
  const resolved = await resolveDecisionFromChatKit(input, body.tools, body.tool_choice, env, upstreamFetch);
  if (!resolved.ok) {
    return resolved.response;
  }

  const response = responsesApiFromDecision(resolved.decision, model);

  if (body.stream) {
    return responsesApiStream(response);
  }

  return json(response);
}

async function resolveDecisionFromChatKit(
  input: ChatKitInput,
  tools: ChatTool[] = [],
  toolChoice: unknown,
  env: ChatKitProviderBindings,
  upstreamFetch: Fetcher
): Promise<ChatKitDecisionResult> {
  const upstream = await callChatKit(input, env, upstreamFetch);
  if (!upstream.ok) {
    return { ok: false, response: await upstreamError(upstream) };
  }

  const read = await readChatKitAssistantText(upstream);
  let resolved = resolveAssistantDecision(read.text, tools, toolChoice);

  if (!resolved.ok && shouldAttemptProtocolRepair(tools, toolChoice)) {
    const repairInput = buildProtocolRepairInput(input, read.text, resolved.message, toolChoice);
    const repairUpstream = await callChatKit(repairInput, env, upstreamFetch);
    if (!repairUpstream.ok) {
      return { ok: false, response: await upstreamError(repairUpstream) };
    }

    const repairRead = await readChatKitAssistantText(repairUpstream);
    const repaired = resolveAssistantDecision(repairRead.text, tools, toolChoice);
    if (repaired.ok) {
      return { ok: true, decision: repaired.decision };
    }

    resolved = {
      ok: false,
      message: `${resolved.message} Protocol repair also failed: ${repaired.message}`
    };
  }

  if (!resolved.ok) {
    return { ok: false, response: jsonError("adapter_protocol_error", resolved.message, 502) };
  }

  return { ok: true, decision: resolved.decision };
}

function shouldAttemptProtocolRepair(tools: ChatTool[], toolChoice: unknown): boolean {
  const normalizedTools = normalizeToolsForPrompt(tools);
  const normalizedChoice = normalizeToolChoice(toolChoice);
  return normalizedTools.length > 0 && normalizedChoice.type !== "none";
}

function buildProtocolRepairInput(
  input: ChatKitInput,
  invalidAssistantText: string,
  errorMessage: string,
  toolChoice: unknown
): ChatKitInput {
  const originalPrompt = input.content[0]?.text || "";
  const normalizedChoice = normalizeToolChoice(toolChoice);
  const invalidPreview = truncateForProtocolRepair(invalidAssistantText);
  const repairPrompt = [
    originalPrompt,
    "",
    "PROTOCOL_REPAIR_REQUEST",
    "Your previous response violated the OpenAI-compatible adapter protocol.",
    `Adapter error: ${errorMessage}`,
    "Rewrite your previous response as exactly one valid terminal block.",
    "Do not apologize. Do not explain the protocol. Do not output prose outside the terminal block.",
    "If the previous response said you would inspect, edit, run, test, search, or continue work, emit a matching <tool_calls> block now.",
    "Previous invalid assistant response, JSON-escaped and possibly truncated:",
    JSON.stringify(invalidPreview),
    finalOutputContract(normalizedChoice, toolChoice)
  ].join("\n");

  return {
    ...input,
    content: [{ type: "input_text", text: repairPrompt }]
  };
}

function truncateForProtocolRepair(text: string): string {
  const trimmed = text.trim();
  if (trimmed.length <= MAX_PROTOCOL_REPAIR_PREVIEW_CHARS) {
    return trimmed;
  }

  return `${trimmed.slice(0, MAX_PROTOCOL_REPAIR_PREVIEW_CHARS)}\n...[truncated]`;
}

function buildPrompt(messages: ChatMessage[], tools: ChatTool[], toolChoice: unknown, instructions?: unknown): string {
  const lines: string[] = [];
  const promptTools = normalizeToolsForPrompt(tools);
  const normalizedChoice = normalizeToolChoice(toolChoice);

  lines.push("You are serving an OpenAI-compatible API adapter backed by ChatKit.");

  const instructionText = instructionsToText(instructions);
  if (instructionText) {
    lines.push("INSTRUCTIONS", instructionText);
  }

  if (promptTools.length > 0 && normalizedChoice.type !== "none") {
    lines.push(
      "OPENAI_COMPAT_TAGGED_TOOL_PROTOCOL",
      "You can request tool calls, but you cannot execute tools yourself.",
      "Return exactly one terminal block and no text outside XML-like tags.",
      "Optional private reasoning may appear only inside <think>...</think> before the terminal block.",
      "To call function tools, return exactly: <tool_calls>[{\"name\":\"tool_name\",\"arguments\":{}}]</tool_calls>",
      "For namespaced function tools, include namespace: {\"namespace\":\"namespace_name\",\"name\":\"tool_name\",\"arguments\":{}}.",
      "For custom tool calls use {\"name\":\"tool_name\",\"input\":\"raw input\"} inside <tool_calls>.",
      "To answer the user, return exactly: <final_answer>your final answer</final_answer>",
      "Never say you will inspect, run, search, or call something next; emit <tool_calls> instead.",
      toolChoiceInstruction(normalizedChoice),
      `tool_choice=${JSON.stringify(toolChoice ?? "auto")}`,
      "## Available tools",
      formatToolsForPrompt(promptTools)
    );
  } else {
    lines.push("Answer the user directly. Do not invent tool calls.");
  }

  lines.push("CONVERSATION");
  for (const message of messages) {
    lines.push(formatMessage(message));
  }

  if (promptTools.length > 0 && normalizedChoice.type !== "none") {
    lines.push(finalOutputContract(normalizedChoice, toolChoice), codexShellToolRules(promptTools));
  }

  return lines.join("\n");
}

function finalOutputContract(choice: NormalizedToolChoice, toolChoice: unknown): string {
  return [
    "FINAL_OUTPUT_CONTRACT",
    "This is a machine protocol, not a normal chat response.",
    "Plain prose outside these tags is a protocol error.",
    "Return exactly one of these terminal blocks and then stop:",
    "- <tool_calls>[{\"name\":\"tool_name\",\"arguments\":{}}]</tool_calls>",
    "- <tool_calls>[{\"name\":\"tool_name\",\"input\":\"raw input\"}]</tool_calls>",
    "- <final_answer>your final answer</final_answer>",
    "If you need to inspect, edit, run, test, search, or continue work, emit <tool_calls>; do not describe the plan.",
    toolChoiceInstruction(choice),
    `tool_choice=${JSON.stringify(toolChoice ?? "auto")}`
  ].join("\n");
}

function codexShellToolRules(tools: PromptTool[]): string {
  const hasShellTool = tools.some((tool) =>
    ["exec", "exec_command", "shell", "bash"].includes(tool.name)
  );

  if (!hasShellTool) {
    return "";
  }

  return [
    "CODEX_SHELL_TOOL_RULES",
    "Confirm the target project directory before creating files or directories.",
    "Use the tool workdir field or an explicit cd into the existing project root.",
    "Do not create sibling src/, data/, types/, or utils/ directories from a parent workspace unless the user asked for a new project.",
    "Do not use echo to write multi-line source files; quoting breaks easily.",
    "Prefer apply_patch when available. If only shell is available, use cat <<'EOF' with a single-quoted delimiter, or a small script that writes exact file contents.",
    "After editing, run a focused build, typecheck, or file readback before claiming success."
  ].join("\n");
}

function formatToolsForPrompt(tools: PromptTool[]): string {
  return tools.map(formatToolForPrompt).join("\n");
}

function formatToolForPrompt(tool: PromptTool): string {
  if (tool.type === "custom") {
    const format = tool.format;
    const formatText =
      format && (format.type || format.syntax)
        ? ` format=${format.type || "unknown"}/${format.syntax || "unknown"}`
        : "";
    return `- ${tool.name}(custom freeform): ${tool.description}${formatText}`;
  }

  const params = parametersSummary(tool.parameters);
  const qualifiedName = tool.namespace ? `${tool.namespace}.${tool.name}` : tool.name;
  return `- ${qualifiedName}(${params}): ${tool.description}`;
}

function parametersSummary(parameters: unknown): string {
  if (!isRecord(parameters)) {
    return "";
  }

  const properties = isRecord(parameters.properties) ? parameters.properties : {};
  const required = Array.isArray(parameters.required) ? new Set(parameters.required.map(String)) : new Set<string>();
  return Object.entries(properties)
    .map(([name, value]) => {
      const type = isRecord(value) && typeof value.type === "string" ? value.type : "any";
      return `${name}: ${type}${required.has(name) ? " (required)" : ""}`;
    })
    .join(", ");
}

function normalizeToolsForPrompt(tools: ChatTool[]): PromptTool[] {
  const normalized: PromptTool[] = [];

  for (const tool of tools) {
    if (tool.type === "namespace") {
      const namespace = nonEmptyString(tool.name);
      const namespaceDescription = nonEmptyString(tool.description) || "";
      for (const nestedTool of Array.isArray(tool.tools) ? tool.tools : []) {
        if (nestedTool.type !== "function") {
          continue;
        }
        const name = nonEmptyString(nestedTool.function?.name) || nonEmptyString(nestedTool.name);
        if (!name) {
          continue;
        }
        normalized.push({
          type: "function",
          namespace,
          name,
          description:
            nonEmptyString(nestedTool.function?.description) ||
            nonEmptyString(nestedTool.description) ||
            namespaceDescription,
          parameters: nestedTool.function?.parameters ?? nestedTool.parameters ?? { type: "object", properties: {} }
        });
      }
      continue;
    }

    if (tool.type === "custom") {
      const name = nonEmptyString(tool.name);
      if (!name) {
        continue;
      }
      normalized.push({
        type: "custom",
        name,
        description: nonEmptyString(tool.description) || "",
        format: tool.format
      });
      continue;
    }

    if (tool.type !== "function") {
      continue;
    }

    const name = nonEmptyString(tool.function?.name) || nonEmptyString(tool.name);
    if (!name) {
      continue;
    }
    normalized.push({
      type: "function",
      name,
      description: nonEmptyString(tool.function?.description) || nonEmptyString(tool.description) || "",
      parameters: tool.function?.parameters ?? tool.parameters ?? { type: "object", properties: {} }
    });
  }

  return normalized;
}

function toolChoiceInstruction(choice: NormalizedToolChoice): string {
  switch (choice.type) {
    case "required":
      return "tool_choice requires at least one <tool_calls> block before any <final_answer>.";
    case "function":
      return `tool_choice requires calling only ${JSON.stringify(choice.name)} when a tool call is needed.`;
    case "none":
      return "tool_choice forbids tool calls.";
    default:
      return "tool_choice allows either <tool_calls> or <final_answer>.";
  }
}

function normalizeToolChoice(toolChoice: unknown): NormalizedToolChoice {
  if (toolChoice === "required" || toolChoice === "none" || toolChoice === "auto") {
    return { type: toolChoice };
  }

  if (isRecord(toolChoice) && toolChoice.type === "function") {
    const name = nonEmptyString(toolChoice.name);
    if (name) {
      return { type: "function", name };
    }

    const func = isRecord(toolChoice.function) ? toolChoice.function : undefined;
    const functionName = nonEmptyString(func?.name);
    if (functionName) {
      return { type: "function", name: functionName };
    }
  }

  return { type: "auto" };
}

function validateDecisionAgainstTools(
  decision: AssistantDecision,
  tools: PromptTool[],
  choice: NormalizedToolChoice
): string | undefined {
  if (choice.type === "none" && decision.type === "tool_calls") {
    return "tool_choice is none, but the assistant emitted tool calls.";
  }

  if ((choice.type === "required" || choice.type === "function") && decision.type === "final") {
    return "tool_choice requires a tool call, but the assistant emitted a final answer.";
  }

  if (decision.type !== "tool_calls") {
    return undefined;
  }

  const allowedNames = new Set(tools.map((tool) => tool.name));
  for (const call of decision.tool_calls) {
    if (allowedNames.size > 0 && !allowedNames.has(call.name)) {
      return `Assistant requested unknown tool: ${call.name}`;
    }

    if (choice.type === "function" && call.name !== choice.name) {
      return `tool_choice requires ${choice.name}, but the assistant requested ${call.name}.`;
    }

    const tool = findPromptToolForCall(tools, call);
    if (tool?.type === "custom" && typeof call.input !== "string") {
      return `Custom tool ${call.name} requires a string input.`;
    }
  }

  return undefined;
}

function formatMessage(message: ChatMessage): string {
  const role = nonEmptyString(message.role) || "user";
  const segments = [`<message role=${JSON.stringify(role)}`];

  if (message.type) {
    segments.push(` type=${JSON.stringify(message.type)}`);
  }
  if (message.id) {
    segments.push(` id=${JSON.stringify(message.id)}`);
  }
  if (message.call_id) {
    segments.push(` call_id=${JSON.stringify(message.call_id)}`);
  }
  if (message.name) {
    segments.push(` name=${JSON.stringify(message.name)}`);
  }

  if (message.tool_call_id) {
    segments.push(` tool_call_id=${JSON.stringify(message.tool_call_id)}`);
  }

  segments.push(">");

  segments.push(`\n${messageBodyToProtocolText(message, role)}\n</message>`);
  return segments.join("");
}

function messageBodyToProtocolText(message: ChatMessage, role: string): string {
  if (role === "assistant" && message.type === "function_call") {
    return `<tool_calls>${JSON.stringify([functionCallMessageToProtocol(message)])}</tool_calls>`;
  }

  if (role === "assistant" && message.type === "custom_tool_call") {
    return `<tool_calls>${JSON.stringify([customToolCallMessageToProtocol(message)])}</tool_calls>`;
  }

  if (role === "assistant" && message.tool_calls) {
    const calls = protocolCallsFromOpenAiToolCalls(message.tool_calls);
    if (calls.length > 0) {
      return `<tool_calls>${JSON.stringify(calls)}</tool_calls>`;
    }
  }

  if (role === "tool" || message.type === "function_call_output") {
    const callId = nonEmptyString(message.tool_call_id) || nonEmptyString(message.call_id) || nonEmptyString(message.id);
    const prefix = callId ? `Tool result for call_id=${callId}` : "Tool result";
    return `${prefix}\n<tool_result>${contentToText(message.content)}</tool_result>\n\n${toolResultFollowupInstruction()}`;
  }

  return contentToText(message.content);
}

function toolResultFollowupInstruction(): string {
  return [
    "Now output exactly one response using only the tagged protocol:",
    "- optional <think>...</think>",
    "- then exactly one <tool_calls>[...]</tool_calls> or <final_answer>...</final_answer>",
    "Do not output Observation.",
    "Do not output <tool_result>.",
    "Stop immediately after </tool_calls> or </final_answer>."
  ].join("\n");
}

function functionCallMessageToProtocol(message: ChatMessage): ToolCallIntent {
  const callId = nonEmptyString(message.call_id) || nonEmptyString(message.id);
  return {
    id: callId,
    name: nonEmptyString(message.name) || "unknown_tool",
    arguments: parseJsonValueOrString(contentToText(message.content).trim(), {})
  };
}

function customToolCallMessageToProtocol(message: ChatMessage): ToolCallIntent {
  const callId = nonEmptyString(message.call_id) || nonEmptyString(message.id);
  return {
    id: callId,
    name: nonEmptyString(message.name) || "unknown_tool",
    input: contentToText(message.content),
    kind: "custom"
  };
}

function protocolCallsFromOpenAiToolCalls(value: unknown): ToolCallIntent[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((call) => protocolCallFromOpenAiToolCall(call))
    .filter((call): call is ToolCallIntent => call !== undefined);
}

function protocolCallFromOpenAiToolCall(value: unknown): ToolCallIntent | undefined {
  if (!isRecord(value)) {
    return undefined;
  }

  const directName = nonEmptyString(value.name);
  const directArgs = value.arguments;
  if (directName) {
    return {
      id: nonEmptyString(value.id),
      name: directName,
      arguments: directArgs ?? {}
    };
  }

  const func = isRecord(value.function) ? value.function : undefined;
  const functionName = nonEmptyString(func?.name);
  if (!functionName) {
    return undefined;
  }

  return {
    id: nonEmptyString(value.id),
    name: functionName,
    arguments: parseJsonValueOrString(func?.arguments, {})
  };
}

function contentToText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        if (isRecord(part) && typeof part.text === "string") {
          return part.text;
        }
        return JSON.stringify(part);
      })
      .join("\n");
  }

  if (content === undefined || content === null) {
    return "";
  }

  return JSON.stringify(content);
}

function functionOutputToText(output: unknown): string {
  if (typeof output === "string") {
    return output;
  }

  if (Array.isArray(output)) {
    return contentToText(output);
  }

  if (isRecord(output)) {
    if (typeof output.content === "string") {
      return output.content;
    }
    if (typeof output.text === "string") {
      return output.text;
    }
    if (Array.isArray(output.content_items)) {
      return contentToText(output.content_items);
    }
    if (Array.isArray(output.content)) {
      return contentToText(output.content);
    }
  }

  return contentToText(output);
}

function responsesInputToMessages(input: unknown): ChatMessage[] {
  if (typeof input === "string") {
    return [{ role: "user", content: input }];
  }

  if (Array.isArray(input)) {
    return input.map(responseInputItemToMessage);
  }

  return [{ role: "user", content: input ?? "" }];
}

function responseInputItemToMessage(item: unknown): ChatMessage {
  if (!isRecord(item)) {
    return { role: "user", content: item };
  }

  const type = nonEmptyString(item.type);
  const id = nonEmptyString(item.id);
  const callId = nonEmptyString(item.call_id);

  if (type === "message") {
    return {
      role: nonEmptyString(item.role) || "user",
      type,
      id,
      content: item.content ?? ""
    };
  }

  if (type === "function_call") {
    return {
      role: "assistant",
      type,
      id,
      call_id: callId,
      name: nonEmptyString(item.name),
      content: item.arguments ?? ""
    };
  }

  if (type === "custom_tool_call") {
    return {
      role: "assistant",
      type,
      id,
      call_id: callId,
      name: nonEmptyString(item.name),
      content: item.input ?? ""
    };
  }

  if (type === "function_call_output") {
    return {
      role: "tool",
      type,
      id,
      call_id: callId,
      content: functionOutputToText(item.output ?? item.content ?? "")
    };
  }

  if (type === "custom_tool_call_output") {
    return {
      role: "tool",
      type,
      id,
      call_id: callId,
      name: nonEmptyString(item.name),
      content: functionOutputToText(item.output ?? item.content ?? "")
    };
  }

  if (typeof item.role === "string") {
    return {
      role: item.role,
      type,
      id,
      content: item.content ?? item
    };
  }

  return {
    role: "user",
    type,
    id,
    content: item.content ?? item
  };
}

function instructionsToText(instructions: unknown): string {
  if (instructions === undefined || instructions === null) {
    return "";
  }

  return contentToText(instructions).trim();
}

function supportedModels(env: ChatKitProviderBindings): string[] {
  const configured = env.MODEL_LIST?.split(",").map((model) => model.trim()).filter(Boolean);
  return configured && configured.length > 0 ? configured : DEFAULT_MODELS;
}

function validateModel(model: string, env: ChatKitProviderBindings): Response | undefined {
  if (supportedModels(env).includes(model)) {
    return undefined;
  }

  return jsonError("invalid_model", `Unsupported model: ${model}`, 400);
}

function codexModelInfo(slug: string) {
  return {
    id: slug,
    slug,
    model: slug,
    display_name: slug,
    description: modelDescription(slug),
    default_reasoning_level: "medium",
    supported_reasoning_levels: [
      { effort: "low", description: "Fast" },
      { effort: "medium", description: "Balanced" },
      { effort: "high", description: "More reasoning" }
    ],
    shell_type: "shell_command",
    visibility: "list",
    supported_in_api: true,
    priority: modelPriority(slug),
    upgrade: null,
    base_instructions: "You are Codex, a pragmatic software engineering agent.",
    supports_reasoning_summaries: false,
    support_verbosity: true,
    default_verbosity: "medium",
    apply_patch_tool_type: "freeform",
    truncation_policy: { mode: "tokens", limit: 128000 },
    supports_parallel_tool_calls: true,
    supports_image_detail_original: false,
    context_window: 128000,
    experimental_supported_tools: [],
    use_responses_lite: true
  };
}

function modelDescription(slug: string): string {
  switch (slug) {
    case "gpt-5-nano":
      return "Fast answers";
    case "gpt-5-reason":
      return "Tuned for reasoning and logic";
    case "gpt-5-pro":
      return "Research grade intelligence";
    default:
      return "Balanced intelligence";
  }
}

function modelPriority(slug: string): number {
  switch (slug) {
    case "gpt-5":
      return 100;
    case "gpt-5-reason":
      return 90;
    case "gpt-5-nano":
      return 80;
    case "gpt-5-pro":
      return 70;
    default:
      return 0;
  }
}

async function callChatKit(
  input: ChatKitInput,
  env: ChatKitProviderBindings,
  upstreamFetch: Fetcher
): Promise<Response> {
  const upstreamUrl = env.CHATKIT_UPSTREAM_URL || DEFAULT_CHATKIT_UPSTREAM_URL;
  const origin = env.CHATKIT_ORIGIN || DEFAULT_CHATKIT_ORIGIN;
  const referer = env.CHATKIT_REFERER || `${origin}/`;

  return upstreamFetch(upstreamUrl, {
    method: "POST",
    headers: {
      accept: "text/event-stream",
      "content-type": "application/json",
      origin,
      referer
    },
    body: JSON.stringify({
      type: "threads.create",
      params: { input }
    })
  });
}

async function readChatKitAssistantText(response: Response): Promise<ChatKitReadResult> {
  if (!response.body) {
    return { text: "" };
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let textBytes = 0;
  let text = "";
  let finalText = "";
  let threadId: string | undefined;

  while (true) {
    const { done, value } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    if (buffer.length > MAX_CHATKIT_SSE_RECORD_BYTES) {
      await reader.cancel("ChatKit SSE record exceeded maximum size.");
      throw new Error("ChatKit SSE record exceeded maximum size.");
    }

    const records = splitCompleteSseRecords(buffer);
    buffer = records.remainder;

    for (const record of records.complete) {
      const result = consumeChatKitSseRecord(record, { text, textBytes, finalText, threadId });
      text = result.text;
      textBytes = result.textBytes;
      finalText = result.finalText;
      threadId = result.threadId;

      if (textBytes > MAX_CHATKIT_TEXT_BYTES) {
        await reader.cancel("ChatKit assistant text exceeded maximum size.");
        throw new Error("ChatKit assistant text exceeded maximum size.");
      }

      if (hasCompleteTerminalBlock(text)) {
        await reader.cancel("OpenAI-compatible terminal block received.");
        return { text, threadId };
      }
    }
  }

  buffer += decoder.decode();
  if (buffer) {
    const result = consumeChatKitSseRecord(buffer, { text, textBytes, finalText, threadId });
    text = result.text;
    textBytes = result.textBytes;
    finalText = result.finalText;
    threadId = result.threadId;

    if (textBytes > MAX_CHATKIT_TEXT_BYTES) {
      throw new Error("ChatKit assistant text exceeded maximum size.");
    }
  }

  return { text: text || finalText, threadId };
}

type ChatKitSseAccumulator = {
  text: string;
  textBytes: number;
  finalText: string;
  threadId?: string;
};

function splitCompleteSseRecords(buffer: string): { complete: string[]; remainder: string } {
  const complete: string[] = [];
  let start = 0;

  while (true) {
    const lfLf = buffer.indexOf("\n\n", start);
    const crLfCrLf = buffer.indexOf("\r\n\r\n", start);
    const indexes = [lfLf, crLfCrLf].filter((index) => index >= 0);
    if (indexes.length === 0) {
      break;
    }

    const end = Math.min(...indexes);
    const separatorLength = buffer.startsWith("\r\n\r\n", end) ? 4 : 2;
    complete.push(buffer.slice(start, end));
    start = end + separatorLength;
  }

  return { complete, remainder: buffer.slice(start) };
}

function consumeChatKitSseRecord(record: string, state: ChatKitSseAccumulator): ChatKitSseAccumulator {
  const event = parseSseDataRecord(record);
  if (!event) {
    return state;
  }

  if (event.type === "thread.created" && isRecord(event.thread) && typeof event.thread.id === "string") {
    return { ...state, threadId: event.thread.id };
  }

  if (event.type === "thread.item.updated") {
    const update = isRecord(event.update) ? event.update : undefined;
    if (update?.type === "assistant_message.content_part.text_delta" && typeof update.delta === "string") {
      return {
        ...state,
        text: state.text + update.delta,
        textBytes: state.textBytes + new TextEncoder().encode(update.delta).byteLength
      };
    }
    if (
      update?.type === "assistant_message.content_part.done" &&
      isRecord(update.content) &&
      typeof update.content.text === "string"
    ) {
      return { ...state, finalText: update.content.text };
    }
  }

  if (event.type === "thread.item.done" && isRecord(event.item) && event.item.type === "assistant_message") {
    const itemContent = event.item.content;
    if (Array.isArray(itemContent)) {
      const outputText = itemContent
        .map((part) => (isRecord(part) && typeof part.text === "string" ? part.text : ""))
        .join("");
      if (outputText) {
        return { ...state, finalText: outputText };
      }
    }
  }

  return state;
}

function parseSseDataRecord(record: string): Record<string, unknown> | undefined {
  const data = record
    .split(/\r?\n/)
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart())
    .join("\n")
    .trim();

  if (!data || data === "[DONE]") {
    return undefined;
  }

  try {
    const parsed = JSON.parse(data);
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function hasCompleteTerminalBlock(text: string): boolean {
  const normalized = stripLeadingThinkBlocks(unwrapMarkdownFence(text.trim())).trim();
  return normalized.endsWith("</tool_calls>") || normalized.endsWith("</final_answer>");
}

function chatCompletionFromDecision(decision: AssistantDecision, model: string) {
  const id = `chatcmpl_${crypto.randomUUID()}`;
  const created = unixNow();

  if (decision.type === "tool_calls") {
    return {
      id,
      object: "chat.completion",
      created,
      model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            tool_calls: decision.tool_calls.map(openAiToolCall)
          },
          finish_reason: "tool_calls"
        }
      ],
      usage: zeroUsage()
    };
  }

  return {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content: decision.content },
        finish_reason: "stop"
      }
    ],
    usage: zeroUsage()
  };
}

function responsesApiFromDecision(decision: AssistantDecision, model: string) {
  const id = `resp_${crypto.randomUUID()}`;
  const createdAt = unixNow();

  if (decision.type === "tool_calls") {
    return {
      id,
      object: "response",
      created_at: createdAt,
      status: "completed",
      model,
      output: decision.tool_calls.map(responsesToolCallItem),
      output_text: "",
      usage: responsesUsage()
    };
  }

  return {
    id,
    object: "response",
    created_at: createdAt,
    status: "completed",
    model,
    output: [
      {
        id: `msg_${crypto.randomUUID()}`,
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: decision.content, annotations: [] }]
      }
    ],
    output_text: decision.content,
    usage: responsesUsage()
  };
}

function responsesToolCallItem(call: ToolCallIntent) {
  if (call.kind === "custom" || call.input !== undefined) {
    const id = call.id || `call_${crypto.randomUUID().replaceAll("-", "").slice(0, 24)}`;
    return {
      id,
      type: "custom_tool_call",
      call_id: id,
      name: call.name,
      input: call.input ?? stringifyArguments(call.arguments)
    };
  }

  const openAiCall = openAiToolCall(call);
  return {
    id: openAiCall.id,
    type: "function_call",
    call_id: openAiCall.id,
    name: openAiCall.function.name,
    namespace: call.namespace,
    arguments: openAiCall.function.arguments
  };
}

function openAiToolCall(call: ToolCallIntent) {
  return {
    id: call.id || `call_${crypto.randomUUID().replaceAll("-", "").slice(0, 24)}`,
    type: "function",
    function: {
      name: call.name,
      arguments: stringifyArguments(call.arguments)
    }
  };
}

function stringifyArguments(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }

  if (value === undefined) {
    return "{}";
  }

  return JSON.stringify(value);
}

function normalizeToolCall(value: unknown): ToolCallIntent | undefined {
  if (!isRecord(value) || typeof value.name !== "string" || !value.name) {
    return undefined;
  }

  const input = typeof value.input === "string" ? value.input : undefined;
  return {
    id: typeof value.id === "string" ? value.id : undefined,
    namespace: typeof value.namespace === "string" && value.namespace ? value.namespace : undefined,
    name: value.name,
    arguments: input === undefined ? value.arguments ?? {} : value.arguments,
    input
  };
}

function unwrapMarkdownFence(text: string): string {
  const fenceMatch = text.match(/^```(?:json)?\s*([\s\S]*?)\s*```$/);
  if (fenceMatch) {
    return fenceMatch[1].trim();
  }
  return text;
}

function parseJsonObject(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    const extracted = extractFirstJsonObject(text);
    if (!extracted) {
      return undefined;
    }
    try {
      return JSON.parse(extracted);
    } catch {
      return undefined;
    }
  }
}

function parseJsonArray(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function parseJsonValueOrString(value: unknown, fallback: unknown): unknown {
  if (typeof value !== "string") {
    return value ?? fallback;
  }

  if (!value.trim()) {
    return fallback;
  }

  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
}

function extractFirstJsonObject(text: string): string | undefined {
  const start = text.indexOf("{");
  if (start < 0) {
    return undefined;
  }

  let depth = 0;
  let inString = false;
  let escaped = false;

  for (let index = start; index < text.length; index += 1) {
    const char = text[index];

    if (inString) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === '"') {
        inString = false;
      }
      continue;
    }

    if (char === '"') {
      inString = true;
      continue;
    }

    if (char === "{") {
      depth += 1;
    } else if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return text.slice(start, index + 1);
      }
    }
  }

  return undefined;
}

function chatCompletionStream(completion: ReturnType<typeof chatCompletionFromDecision>): Response {
  const choice = completion.choices[0];
  const chunks: string[] = [];

  chunks.push(
    sse({
      id: completion.id,
      object: "chat.completion.chunk",
      created: completion.created,
      model: completion.model,
      choices: [{ index: 0, delta: { role: "assistant" }, finish_reason: null }]
    })
  );

  if ("tool_calls" in choice.message && Array.isArray(choice.message.tool_calls)) {
    chunks.push(
      sse({
        id: completion.id,
        object: "chat.completion.chunk",
        created: completion.created,
        model: completion.model,
        choices: [{ index: 0, delta: { tool_calls: choice.message.tool_calls }, finish_reason: null }]
      })
    );
  } else {
    chunks.push(
      sse({
        id: completion.id,
        object: "chat.completion.chunk",
        created: completion.created,
        model: completion.model,
        choices: [{ index: 0, delta: { content: choice.message.content }, finish_reason: null }]
      })
    );
  }

  chunks.push(
    sse({
      id: completion.id,
      object: "chat.completion.chunk",
      created: completion.created,
      model: completion.model,
      choices: [{ index: 0, delta: {}, finish_reason: choice.finish_reason }]
    }),
    "data: [DONE]\n\n"
  );

  return eventStream(chunks.join(""));
}

function responsesApiStream(response: ReturnType<typeof responsesApiFromDecision>): Response {
  const chunks: string[] = [];

  chunks.push(sse({ type: "response.created", response: { ...response, status: "in_progress", output: [] } }));

  for (let index = 0; index < response.output.length; index += 1) {
    const item = response.output[index];
    if (item.type === "function_call") {
      chunks.push(...responsesFunctionCallEvents(item, index));
    } else if (item.type === "custom_tool_call") {
      chunks.push(...responsesCustomToolCallEvents(item, index));
    } else if (item.type === "message") {
      chunks.push(...responsesMessageEvents(item, index));
    }
  }

  chunks.push(sse({ type: "response.completed", response }), "data: [DONE]\n\n");
  return eventStream(chunks.join(""));
}

function responsesMessageEvents(item: Record<string, unknown>, outputIndex: number): string[] {
  const itemId = nonEmptyString(item.id) || `msg_${crypto.randomUUID()}`;
  const content = Array.isArray(item.content) ? item.content : [];
  const firstPart = content.find((part) => isRecord(part) && part.type === "output_text");
  const text = isRecord(firstPart) && typeof firstPart.text === "string" ? firstPart.text : "";
  const addedItem = { id: itemId, type: "message", status: "in_progress", role: "assistant", content: [] };
  const doneItem = {
    id: itemId,
    type: "message",
    status: "completed",
    role: "assistant",
    content: [{ type: "output_text", text, annotations: [], logprobs: [] }]
  };

  return [
    sse({
      type: "response.output_item.added",
      sequence_number: 0,
      output_index: outputIndex,
      item: addedItem
    }),
    sse({
      type: "response.content_part.added",
      sequence_number: 0,
      item_id: itemId,
      output_index: outputIndex,
      content_index: 0,
      part: { type: "output_text", text: "", annotations: [], logprobs: [] }
    }),
    sse({
      type: "response.output_text.delta",
      sequence_number: 0,
      item_id: itemId,
      output_index: outputIndex,
      content_index: 0,
      delta: text,
      logprobs: []
    }),
    sse({
      type: "response.output_text.done",
      sequence_number: 0,
      item_id: itemId,
      output_index: outputIndex,
      content_index: 0,
      text,
      logprobs: []
    }),
    sse({
      type: "response.content_part.done",
      sequence_number: 0,
      item_id: itemId,
      output_index: outputIndex,
      content_index: 0,
      part: { type: "output_text", text, annotations: [], logprobs: [] }
    }),
    sse({
      type: "response.output_item.done",
      sequence_number: 0,
      output_index: outputIndex,
      item: doneItem
    })
  ];
}

function responsesFunctionCallEvents(item: Record<string, unknown>, outputIndex: number): string[] {
  const itemId = nonEmptyString(item.id) || `fc_${crypto.randomUUID()}`;
  const callId = nonEmptyString(item.call_id) || itemId;
  const name = nonEmptyString(item.name) || "unknown_tool";
  const argumentsText = typeof item.arguments === "string" ? item.arguments : "{}";
  const addedItem = {
    id: itemId,
    type: "function_call",
    status: "in_progress",
    arguments: "",
    call_id: callId,
    name
  };
  const doneItem = {
    id: itemId,
    type: "function_call",
    status: "completed",
    arguments: argumentsText,
    call_id: callId,
    name
  };

  return [
    sse({
      type: "response.output_item.added",
      sequence_number: 0,
      output_index: outputIndex,
      item: addedItem
    }),
    sse({
      type: "response.function_call_arguments.delta",
      sequence_number: 0,
      item_id: itemId,
      output_index: outputIndex,
      delta: argumentsText
    }),
    sse({
      type: "response.function_call_arguments.done",
      sequence_number: 0,
      item_id: itemId,
      output_index: outputIndex,
      arguments: argumentsText
    }),
    sse({
      type: "response.output_item.done",
      sequence_number: 0,
      output_index: outputIndex,
      item: doneItem
    })
  ];
}

function responsesCustomToolCallEvents(item: Record<string, unknown>, outputIndex: number): string[] {
  const itemId = nonEmptyString(item.id) || `ctc_${crypto.randomUUID()}`;
  const callId = nonEmptyString(item.call_id) || itemId;
  const name = nonEmptyString(item.name) || "unknown_tool";
  const input = typeof item.input === "string" ? item.input : "";
  const addedItem = {
    id: itemId,
    type: "custom_tool_call",
    status: "in_progress",
    call_id: callId,
    name,
    input: ""
  };
  const doneItem = {
    id: itemId,
    type: "custom_tool_call",
    status: "completed",
    call_id: callId,
    name,
    input
  };

  return [
    sse({
      type: "response.output_item.added",
      sequence_number: 0,
      output_index: outputIndex,
      item: addedItem
    }),
    sse({
      type: "response.custom_tool_call_input.delta",
      sequence_number: 0,
      item_id: itemId,
      call_id: callId,
      output_index: outputIndex,
      delta: input
    }),
    sse({
      type: "response.output_item.done",
      sequence_number: 0,
      output_index: outputIndex,
      item: doneItem
    })
  ];
}

function sse(value: unknown): string {
  return `data: ${JSON.stringify(value)}\n\n`;
}

function eventStream(body: string): Response {
  return cors(
    new Response(body, {
      headers: {
        "content-type": "text/event-stream; charset=utf-8",
        "cache-control": "no-cache"
      }
    })
  );
}

function zeroUsage() {
  return {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0
  };
}

function responsesUsage() {
  return {
    input_tokens: 0,
    input_tokens_details: { cached_tokens: 0 },
    output_tokens: 0,
    output_tokens_details: { reasoning_tokens: 0 },
    total_tokens: 0
  };
}

async function upstreamError(response: Response): Promise<Response> {
  const detail = await boundedText(response, 4096);
  return jsonError("upstream_error", detail || `ChatKit upstream returned ${response.status}.`, 502);
}

async function boundedText(response: Response, limit: number): Promise<string> {
  if (!response.body) {
    return "";
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let text = "";

  while (text.length < limit) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    text += decoder.decode(value, { stream: true });
  }

  return text.slice(0, limit);
}

async function isAuthorized(request: Request, env: ChatKitProviderBindings): Promise<boolean> {
  if (!env.PROVIDER_API_KEY) {
    return true;
  }

  const header = request.headers.get("authorization") || "";
  const token = header.startsWith("Bearer ") ? header.slice("Bearer ".length) : "";
  return timingSafeEqual(token, env.PROVIDER_API_KEY);
}

async function timingSafeEqual(a: string, b: string): Promise<boolean> {
  const encoder = new TextEncoder();
  const left = encoder.encode(a);
  const right = encoder.encode(b);

  if (left.length !== right.length) {
    await crypto.subtle.digest("SHA-256", left);
    return false;
  }

  let diff = 0;
  for (let index = 0; index < left.length; index += 1) {
    diff |= left[index] ^ right[index];
  }

  return diff === 0;
}

function json(value: unknown, status = 200): Response {
  return cors(
    new Response(JSON.stringify(value), {
      status,
      headers: { "content-type": "application/json; charset=utf-8" }
    })
  );
}

function jsonError(code: string, message: string, status: number): Response {
  return json({ error: { message, type: code, code } }, status);
}

function cors(response: Response): Response {
  const headers = new Headers(response.headers);
  headers.set("access-control-allow-origin", "*");
  headers.set("access-control-allow-methods", "GET, POST, OPTIONS");
  headers.set("access-control-allow-headers", "authorization, content-type");
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

function unixNow(): number {
  return Math.floor(Date.now() / 1000);
}

function nonEmptyString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export default createChatKitProvider();
