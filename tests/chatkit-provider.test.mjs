import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

const buildDir = mkdtempSync(join(tmpdir(), "chatkit-provider-test-"));

test.after(() => {
  rmSync(buildDir, { force: true, recursive: true });
});

function compileProvider() {
  execFileSync(
    "npx",
    ["tsc", "--outDir", buildDir, "--noEmit", "false", "--module", "ES2022"],
    { cwd: process.cwd(), stdio: "pipe" }
  );
}

async function loadProvider() {
  compileProvider();
  return import(join(buildDir, "index.js"));
}

test("converts Chat Completions tools into a tagged ChatKit prompt protocol", async () => {
  const { buildChatKitInput } = await loadProvider();
  const input = buildChatKitInput({
    model: "gpt-5",
    messages: [{ role: "user", content: "weather in Shanghai?" }],
    tools: [
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather.",
          parameters: {
            type: "object",
            properties: { city: { type: "string" } },
            required: ["city"]
          }
        }
      }
    ]
  });

  assert.equal(input.inference_options.model, "gpt-5");
  assert.match(input.content[0].text, /OPENAI_COMPAT_TAGGED_TOOL_PROTOCOL/);
  assert.match(input.content[0].text, /get_weather/);
  assert.match(input.content[0].text, /weather in Shanghai/);
  assert.match(input.content[0].text, /<tool_calls>/);
  assert.match(input.content[0].text, /<final_answer>/);
  assert.doesNotMatch(input.content[0].text, /"type":"tool_calls"/);
});

test("converts Responses API function tools into a tagged ChatKit prompt protocol", async () => {
  const { buildChatKitInput } = await loadProvider();
  const input = buildChatKitInput({
    model: "gpt-5-nano",
    input: "weather in Shanghai?",
    tools: [
      {
        type: "function",
        name: "get_weather",
        description: "Get weather.",
        parameters: {
          type: "object",
          properties: { city: { type: "string" } },
          required: ["city"]
        }
      }
    ]
  });

  assert.equal(input.inference_options.model, "gpt-5-nano");
  assert.match(input.content[0].text, /OPENAI_COMPAT_TAGGED_TOOL_PROTOCOL/);
  assert.match(input.content[0].text, /get_weather/);
  assert.match(input.content[0].text, /weather in Shanghai/);
});

test("converts Codex custom tools into a tagged ChatKit prompt protocol", async () => {
  const { buildChatKitInput } = await loadProvider();
  const input = buildChatKitInput({
    model: "gpt-5-nano",
    input: "Inspect this repo.",
    tools: [
      {
        type: "custom",
        name: "exec",
        description: "Run JavaScript code that can call local tools.",
        format: {
          type: "grammar",
          syntax: "lark",
          definition: "start: /[\\s\\S]+/"
        }
      }
    ]
  });
  const text = input.content[0].text;

  assert.match(text, /OPENAI_COMPAT_TAGGED_TOOL_PROTOCOL/);
  assert.match(text, /custom tool calls use \{"name":"tool_name","input":"raw input"\}/);
  assert.match(text, /exec\(custom freeform\): Run JavaScript code that can call local tools\./);
  assert.match(text, /format=grammar\/lark/);
  assert.match(text, /Inspect this repo\./);
});

test("preserves Responses instructions and tool-result items in the ChatKit prompt", async () => {
  const { buildChatKitInput } = await loadProvider();
  const input = buildChatKitInput({
    model: "gpt-5-nano",
    instructions: "Persist until the repository question is fully answered.",
    input: [
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Explore this repo." }]
      },
      {
        type: "function_call",
        call_id: "call_list",
        name: "exec_command",
        arguments: '{"cmd":"ls src"}'
      },
      {
        type: "function_call_output",
        call_id: "call_list",
        output: "index.ts"
      }
    ]
  });
  const text = input.content[0].text;

  assert.match(text, /INSTRUCTIONS\nPersist until the repository question is fully answered\./);
  assert.match(text, /<message role="assistant" type="function_call" call_id="call_list" name="exec_command">/);
  assert.match(text, /<tool_calls>\[\{"id":"call_list","name":"exec_command","arguments":\{"cmd":"ls src"\}\}\]<\/tool_calls>/);
  assert.match(text, /<message role="tool" type="function_call_output" call_id="call_list">/);
  assert.match(text, /<tool_result>index\.ts<\/tool_result>/);
  assert.match(text, /index\.ts/);
  assert.doesNotMatch(text, /<message role="user">\n\{"type":"function_call_output"/);
});

test("preserves Codex custom tool output in the ChatKit prompt", async () => {
  const { buildChatKitInput } = await loadProvider();
  const input = buildChatKitInput({
    model: "gpt-5-nano",
    input: [
      {
        type: "message",
        role: "user",
        content: "List src files."
      },
      {
        type: "custom_tool_call_output",
        call_id: "call_exec",
        name: "exec",
        output: "index.ts"
      }
    ],
    tools: [
      {
        type: "custom",
        name: "exec",
        description: "Run JavaScript code.",
        format: { type: "grammar", syntax: "lark", definition: "start: /[\\s\\S]+/" }
      }
    ]
  });
  const text = input.content[0].text;

  assert.match(text, /<message role="tool" type="custom_tool_call_output" call_id="call_exec" name="exec">/);
  assert.match(text, /Tool result for call_id=call_exec/);
  assert.match(text, /<tool_result>index\.ts<\/tool_result>/);
  assert.match(text, /Now output exactly one response using only the tagged protocol/);
});

test("parses tagged tool-call intent out of ChatKit assistant text", async () => {
  const { parseAssistantDecision } = await loadProvider();
  const decision = parseAssistantDecision(
    '<think>Need current weather.</think>\n<tool_calls>[{"name":"get_weather","arguments":{"city":"Shanghai"}}]</tool_calls>'
  );

  assert.equal(decision.type, "tool_calls");
  assert.equal(decision.tool_calls[0].name, "get_weather");
  assert.deepEqual(decision.tool_calls[0].arguments, { city: "Shanghai" });
});

test("returns Responses API custom_tool_call items for Codex custom tools", async () => {
  const { createChatKitProvider } = await loadProvider();
  const provider = createChatKitProvider({
    fetch: async () =>
      new Response(
        [
          'data: {"type":"thread.item.updated","update":{"type":"assistant_message.content_part.text_delta","delta":"<tool_calls>[{\\"name\\":\\"exec\\",\\"input\\":\\"tools.exec_command({ cmd: \\\\\\"ls src\\\\\\" })\\"}]</tool_calls>"}}',
          ""
        ].join("\n")
      )
  });

  const response = await provider.fetch(
    new Request("https://worker.test/v1/responses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gpt-5-nano",
        input: "List src files.",
        tools: [
          {
            type: "custom",
            name: "exec",
            description: "Run JavaScript code.",
            format: { type: "grammar", syntax: "lark", definition: "start: /[\\s\\S]+/" }
          }
        ]
      })
    }),
    {}
  );
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.equal(body.output[0].type, "custom_tool_call");
  assert.equal(body.output[0].name, "exec");
  assert.equal(body.output[0].input, 'tools.exec_command({ cmd: "ls src" })');
});

test("keeps prior completed turns when a later user turn calls a custom tool", async () => {
  const { createChatKitProvider } = await loadProvider();
  let upstreamPrompt = "";
  const provider = createChatKitProvider({
    fetch: async (_url, init) => {
      const body = JSON.parse(init.body);
      upstreamPrompt = body.params.input.content[0].text;
      return new Response(
        [
          'data: {"type":"thread.item.updated","update":{"type":"assistant_message.content_part.text_delta","delta":"<tool_calls>[{\\"name\\":\\"exec\\",\\"input\\":\\"tools.exec_command({ cmd: \\\\\\"ls tests\\\\\\" })\\"}]</tool_calls>"}}',
          ""
        ].join("\n")
      );
    }
  });

  const response = await provider.fetch(
    new Request("https://worker.test/v1/responses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gpt-5-nano",
        instructions: "Use the exec custom tool whenever the user asks to inspect files.",
        input: [
          { type: "message", role: "user", content: "What files are in src?" },
          {
            type: "custom_tool_call",
            call_id: "call_src",
            name: "exec",
            input: 'tools.exec_command({ cmd: "ls src" })'
          },
          {
            type: "custom_tool_call_output",
            call_id: "call_src",
            name: "exec",
            output: "index.ts"
          },
          {
            type: "message",
            role: "assistant",
            content: "The src directory contains index.ts."
          },
          {
            type: "message",
            role: "user",
            content: "Now use the exec custom tool to list the tests directory. Do not answer directly."
          }
        ],
        tools: [
          {
            type: "custom",
            name: "exec",
            description: "Run JavaScript code that can call local tools.",
            format: { type: "grammar", syntax: "lark", definition: "start: /[\\s\\S]+/" }
          }
        ]
      })
    }),
    {}
  );
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.match(upstreamPrompt, /What files are in src\?/);
  assert.match(upstreamPrompt, /<tool_calls>\[\{"id":"call_src","name":"exec","input":"tools\.exec_command/);
  assert.match(upstreamPrompt, /<tool_result>index\.ts<\/tool_result>/);
  assert.match(upstreamPrompt, /The src directory contains index\.ts\./);
  assert.match(upstreamPrompt, /Now use the exec custom tool to list the tests directory/);
  assert.equal(body.output[0].type, "custom_tool_call");
  assert.equal(body.output[0].input, 'tools.exec_command({ cmd: "ls tests" })');
});

test("returns OpenAI Chat Completions tool_calls without executing tools", async () => {
  const { createChatKitProvider } = await loadProvider();
  const provider = createChatKitProvider({
    fetch: async () =>
      new Response(
        [
          'data: {"type":"thread.item.updated","update":{"type":"assistant_message.content_part.text_delta","delta":"<tool_calls>[{\\"name\\":\\"get_weather\\",\\"arguments\\":{\\"city\\":\\"Shanghai\\"}}]</tool_calls>"}}',
          ""
        ].join("\n")
      )
  });

  const response = await provider.fetch(
    new Request("https://worker.test/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gpt-5",
        messages: [{ role: "user", content: "weather in Shanghai?" }],
        tools: [
          {
            type: "function",
            function: {
              name: "get_weather",
              parameters: { type: "object", properties: { city: { type: "string" } } }
            }
          }
        ]
      })
    }),
    {}
  );
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.equal(body.object, "chat.completion");
  assert.equal(body.choices[0].finish_reason, "tool_calls");
  assert.equal(body.choices[0].message.tool_calls[0].type, "function");
  assert.equal(body.choices[0].message.tool_calls[0].function.name, "get_weather");
  assert.equal(body.choices[0].message.tool_calls[0].function.arguments, '{"city":"Shanghai"}');
});

test("returns final text as an OpenAI Chat Completions assistant message", async () => {
  const { createChatKitProvider } = await loadProvider();
  const provider = createChatKitProvider({
    fetch: async () =>
      new Response(
        [
          'data: {"type":"thread.item.updated","update":{"type":"assistant_message.content_part.text_delta","delta":"<final_answer>hello</final_answer>"}}',
          ""
        ].join("\n")
      )
  });

  const response = await provider.fetch(
    new Request("https://worker.test/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gpt-5",
        messages: [{ role: "user", content: "say hello" }],
        tools: []
      })
    }),
    {}
  );
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.equal(body.choices[0].finish_reason, "stop");
  assert.equal(body.choices[0].message.content, "hello");
});

test("stops reading ChatKit SSE after a complete terminal block", async () => {
  const { createChatKitProvider } = await loadProvider();
  const encoder = new TextEncoder();
  let pullCount = 0;
  let canceled = false;
  const provider = createChatKitProvider({
    fetch: async () =>
      new Response(
        new ReadableStream({
          pull(controller) {
            pullCount += 1;
            if (pullCount === 1) {
              controller.enqueue(
                encoder.encode(
                  [
                    'data: {"type":"thread.item.updated","update":{"type":"assistant_message.content_part.text_delta","delta":"<final_answer>OK</final_answer>"}}',
                    "",
                    ""
                  ].join("\n")
                )
              );
              return;
            }

            controller.enqueue(encoder.encode("data: " + "x".repeat(700 * 1024) + "\n\n"));
            controller.close();
          },
          cancel() {
            canceled = true;
          }
        })
      )
  });

  const response = await provider.fetch(
    new Request("https://worker.test/v1/responses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gpt-5-nano",
        input: "Reply exactly: OK",
        tools: []
      })
    }),
    {}
  );
  const body = await response.json();

  assert.equal(response.status, 200);
  assert.equal(body.output_text, "OK");
  assert.equal(canceled, true);
  assert.ok(pullCount <= 2);
});

test("rejects untagged assistant text when tools are active", async () => {
  const { createChatKitProvider } = await loadProvider();
  const provider = createChatKitProvider({
    fetch: async () =>
      new Response(
        [
          'data: {"type":"thread.item.updated","update":{"type":"assistant_message.content_part.text_delta","delta":"I will inspect the project next."}}',
          ""
        ].join("\n")
      )
  });

  const response = await provider.fetch(
    new Request("https://worker.test/v1/responses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gpt-5-nano",
        input: "Explore this repo.",
        tools: [
          {
            type: "function",
            name: "exec_command",
            parameters: { type: "object", properties: { cmd: { type: "string" } }, required: ["cmd"] }
          }
        ]
      })
    }),
    {}
  );
  const body = await response.json();

  assert.equal(response.status, 502);
  assert.equal(body.error.code, "adapter_protocol_error");
});

test("requires bearer token when PROVIDER_API_KEY is configured", async () => {
  const { createChatKitProvider } = await loadProvider();
  const provider = createChatKitProvider({
    fetch: async () => {
      throw new Error("unauthorized requests should not call upstream");
    }
  });

  const unauthorized = await provider.fetch(new Request("https://worker.test/v1/models"), {
    PROVIDER_API_KEY: "test-key"
  });
  const authorized = await provider.fetch(
    new Request("https://worker.test/v1/models", {
      headers: { authorization: "Bearer test-key" }
    }),
    { PROVIDER_API_KEY: "test-key" }
  );

  assert.equal(unauthorized.status, 401);
  assert.equal((await unauthorized.json()).error.code, "invalid_api_key");
  assert.equal(authorized.status, 200);
});

test("streams Responses API completed events with Responses usage shape", async () => {
  const { createChatKitProvider } = await loadProvider();
  const provider = createChatKitProvider({
    fetch: async () =>
      new Response(
        [
          'data: {"type":"thread.item.updated","update":{"type":"assistant_message.content_part.text_delta","delta":"<final_answer>OK</final_answer>"}}',
          ""
        ].join("\n")
      )
  });

  const response = await provider.fetch(
    new Request("https://worker.test/v1/responses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gpt-5-nano",
        input: "Reply exactly: OK",
        stream: true
      })
    }),
    {}
  );
  const body = await response.text();
  const events = parseSseEvents(body);
  const eventTypes = events.map((event) => event.type);
  const deltaEvent = events.find((event) => event.type === "response.output_text.delta");
  const completedEvent = events.find((event) => event.type === "response.completed");

  assert.equal(response.status, 200);
  assert.deepEqual(eventTypes, [
    "response.created",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.completed"
  ]);
  assert.equal(deltaEvent.delta, "OK");
  assert.equal(completedEvent.response.output_text, "OK");
  assert.equal(completedEvent.response.usage.input_tokens, 0);
  assert.equal(completedEvent.response.usage.output_tokens, 0);
  assert.equal(completedEvent.response.usage.total_tokens, 0);
});

test("streams Responses API function-call events for tool execution clients", async () => {
  const { createChatKitProvider } = await loadProvider();
  const provider = createChatKitProvider({
    fetch: async () =>
      new Response(
        [
          'data: {"type":"thread.item.updated","update":{"type":"assistant_message.content_part.text_delta","delta":"<tool_calls>[{\\"name\\":\\"get_weather\\",\\"arguments\\":{\\"city\\":\\"Shanghai\\"}}]</tool_calls>"}}',
          ""
        ].join("\n")
      )
  });

  const response = await provider.fetch(
    new Request("https://worker.test/v1/responses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gpt-5-nano",
        input: "weather in Shanghai?",
        stream: true,
        tools: [
          {
            type: "function",
            name: "get_weather",
            parameters: { type: "object", properties: { city: { type: "string" } } }
          }
        ]
      })
    }),
    {}
  );
  const body = await response.text();
  const events = parseSseEvents(body);
  const addedEvent = events.find((event) => event.type === "response.output_item.added");
  const deltaEvent = events.find((event) => event.type === "response.function_call_arguments.delta");
  const doneEvent = events.find((event) => event.type === "response.function_call_arguments.done");
  const itemDoneEvent = events.find((event) => event.type === "response.output_item.done");

  assert.equal(response.status, 200);
  assert.equal(addedEvent.item.type, "function_call");
  assert.equal(addedEvent.item.name, "get_weather");
  assert.equal(deltaEvent.delta, '{"city":"Shanghai"}');
  assert.equal(doneEvent.arguments, '{"city":"Shanghai"}');
  assert.equal(itemDoneEvent.item.status, "completed");
});

test("streams Responses API custom tool-call events for Codex clients", async () => {
  const { createChatKitProvider } = await loadProvider();
  const provider = createChatKitProvider({
    fetch: async () =>
      new Response(
        [
          'data: {"type":"thread.item.updated","update":{"type":"assistant_message.content_part.text_delta","delta":"<tool_calls>[{\\"name\\":\\"exec\\",\\"input\\":\\"tools.exec_command({ cmd: \\\\\\"ls src\\\\\\" })\\"}]</tool_calls>"}}',
          ""
        ].join("\n")
      )
  });

  const response = await provider.fetch(
    new Request("https://worker.test/v1/responses", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "gpt-5-nano",
        input: "List src files.",
        stream: true,
        tools: [
          {
            type: "custom",
            name: "exec",
            description: "Run JavaScript code.",
            format: { type: "grammar", syntax: "lark", definition: "start: /[\\s\\S]+/" }
          }
        ]
      })
    }),
    {}
  );
  const body = await response.text();
  const events = parseSseEvents(body);
  const addedEvent = events.find((event) => event.type === "response.output_item.added");
  const deltaEvent = events.find((event) => event.type === "response.custom_tool_call_input.delta");
  const itemDoneEvent = events.find((event) => event.type === "response.output_item.done");

  assert.equal(response.status, 200);
  assert.equal(addedEvent.item.type, "custom_tool_call");
  assert.equal(addedEvent.item.name, "exec");
  assert.equal(deltaEvent.delta, 'tools.exec_command({ cmd: "ls src" })');
  assert.equal(itemDoneEvent.item.type, "custom_tool_call");
  assert.equal(itemDoneEvent.item.input, 'tools.exec_command({ cmd: "ls src" })');
});

test("lists the ChatKit Studio model choices", async () => {
  const { createChatKitProvider } = await loadProvider();
  const provider = createChatKitProvider({
    fetch: async () => {
      throw new Error("models endpoint should not call upstream");
    }
  });

  const response = await provider.fetch(new Request("https://worker.test/v1/models"), {});
  const body = await response.json();
  const modelIds = body.data.map((model) => model.id);

  assert.deepEqual(modelIds, ["gpt-5", "gpt-5-nano", "gpt-5-reason", "gpt-5-pro"]);
  assert.deepEqual(
    body.models.map((model) => model.id),
    ["gpt-5", "gpt-5-nano", "gpt-5-reason", "gpt-5-pro"]
  );
  assert.equal(body.models[0].use_responses_lite, true);
});

function parseSseEvents(body) {
  return body
    .split("\n\n")
    .map((chunk) => chunk.trim())
    .filter((chunk) => chunk.startsWith("data: ") && chunk !== "data: [DONE]")
    .map((chunk) => JSON.parse(chunk.slice("data: ".length)));
}
