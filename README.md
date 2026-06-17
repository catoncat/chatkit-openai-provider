# ChatKit OpenAI Provider

Cloudflare Worker that wraps ChatKit Studio's internal `/chatkit` transport as an OpenAI-compatible API.

This adapter does not run a second LLM and does not execute tools. Tool execution stays on the OpenAI-compatible client side:

1. The client sends OpenAI-style `tools`.
2. The Worker turns those schemas into a tagged ChatKit prompt contract.
3. ChatKit returns a tool-call intent.
4. The Worker converts it back to OpenAI `tool_calls`, `function_call`, or `custom_tool_call` items.
5. The client executes the tool and sends the tool result in the next turn.

## Endpoints

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`

## Models

- `gpt-5`
- `gpt-5-nano`
- `gpt-5-reason`
- `gpt-5-pro`

## Local Development

```bash
npm install
npm run dev
```

Health check:

```bash
curl http://127.0.0.1:8787/health
```

Responses API:

```bash
curl http://127.0.0.1:8787/v1/responses \
  -H 'content-type: application/json' \
  --data '{
    "model": "gpt-5-nano",
    "input": "Reply exactly: pong"
  }'
```

Chat Completions tool call:

```bash
curl http://127.0.0.1:8787/v1/chat/completions \
  -H 'content-type: application/json' \
  --data '{
    "model": "gpt-5-nano",
    "messages": [
      { "role": "user", "content": "What is the weather in Shanghai?" }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather.",
          "parameters": {
            "type": "object",
            "properties": {
              "city": { "type": "string" }
            },
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

## Authentication

Set a Worker secret to require OpenAI-compatible bearer authentication:

```bash
npx wrangler secret put PROVIDER_API_KEY
```

Then call the API with:

```bash
Authorization: Bearer <PROVIDER_API_KEY>
```

The repository intentionally does not include any API key or secret value.

## Deploy

```bash
npm run deploy
```

The default `wrangler.jsonc` routes the Worker to `oai.chen.rs/*`. Change the route before deploying to a different domain.
