// worker.sample.ts
// Cloudflare Worker — AI API Proxy with Provider Fallback Chains
//
// Routes requests to the right LLM/VLM provider based on the X-API-Type header.
//
// Deploy: npx wrangler deploy
// Secrets (set via `npx wrangler secret put <NAME>`):
//   FIREWORKS_VISION_KEY_1   – primary Fireworks vision account       (minimax-m3)
//   FIREWORKS_VISION_KEY_2   – fallback Fireworks vision account      (minimax-m3)
//   GROQ_KEY_1               – primary Groq key                       (openai/gpt-oss-120b)
//   GROQ_KEY_2               – fallback Groq key                      (openai/gpt-oss-120b)
//   CEREBRAS_KEY             – Cerebras API key                       (gpt-oss-120b)
//   FIREWORKS_TEXT_KEY       – Fireworks text account (last resort)    (minimax-m3)

// =========================================================================
// Types
// =========================================================================

interface Env {
  FIREWORKS_VISION_KEY_1: string;
  FIREWORKS_VISION_KEY_2: string;
  GROQ_KEY_1: string;
  GROQ_KEY_2: string;
  CEREBRAS_KEY: string;
  FIREWORKS_TEXT_KEY: string;
}

interface Provider {
  name: string;
  url: string;
  model: string;
  getKey: (env: Env) => string;
  stripReasoningEffort: boolean;
}

interface ChatCompletionBody {
  model: string;
  messages: unknown[];
  max_tokens?: number;
  temperature?: number;
  extra_body?: Record<string, unknown>;
  [key: string]: unknown;
}

// =========================================================================
// Provider chain definitions
// =========================================================================

const VISION_CHAIN: Provider[] = [
  {
    name: "fireworks-vision-1",
    url: "https://api.fireworks.ai/inference/v1/chat/completions",
    model: "accounts/fireworks/models/minimax-m3",
    getKey: (env) => env.FIREWORKS_VISION_KEY_1,
    stripReasoningEffort: false,
  },
  {
    name: "fireworks-vision-2",
    url: "https://api.fireworks.ai/inference/v1/chat/completions",
    model: "accounts/fireworks/models/minimax-m3",
    getKey: (env) => env.FIREWORKS_VISION_KEY_2,
    stripReasoningEffort: false,
  },
];

const TEXT_CHAIN: Provider[] = [
  {
    name: "groq-1",
    url: "https://api.groq.com/openai/v1/chat/completions",
    model: "openai/gpt-oss-120b",
    getKey: (env) => env.GROQ_KEY_1,
    stripReasoningEffort: true,
  },
  {
    name: "groq-2",
    url: "https://api.groq.com/openai/v1/chat/completions",
    model: "openai/gpt-oss-120b",
    getKey: (env) => env.GROQ_KEY_2,
    stripReasoningEffort: true,
  },
  {
    name: "cerebras",
    url: "https://api.cerebras.ai/v1/chat/completions",
    model: "gpt-oss-120b",
    getKey: (env) => env.CEREBRAS_KEY,
    stripReasoningEffort: true,
  },
  {
    name: "fireworks-text",
    url: "https://api.fireworks.ai/inference/v1/chat/completions",
    model: "accounts/fireworks/models/minimax-m3",
    getKey: (env) => env.FIREWORKS_TEXT_KEY,
    stripReasoningEffort: false,
  },
];

// =========================================================================
// Helpers
// =========================================================================

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/**
 * Deep-clone the request body and apply provider-specific overrides.
 *  - Overwrites `model` with the provider's model name.
 *  - Strips `reasoning_effort` from `extra_body` for providers that don't
 *    support it (Groq, Cerebras). Fireworks keeps it.
 */
function cloneBodyForProvider(
  body: ChatCompletionBody,
  provider: Provider
): ChatCompletionBody {
  const cloned: ChatCompletionBody = JSON.parse(JSON.stringify(body));
  cloned.model = provider.model;
  if (provider.stripReasoningEffort && cloned.extra_body?.reasoning_effort) {
    delete cloned.extra_body.reasoning_effort;
    if (Object.keys(cloned.extra_body).length === 0) {
      delete cloned.extra_body;
    }
  }
  return cloned;
}

/**
 * Try a single provider. Returns the Response on success, `null` on a
 * fallback-eligible failure (5xx, 429, timeout, network error), or throws
 * a Response on a hard client error (4xx ≠ 429 — won't be fixed by another
 * provider so we propagate it immediately).
 */
async function tryProvider(
  provider: Provider,
  body: ChatCompletionBody,
  signal: AbortSignal,
  env: Env
): Promise<Response | null> {
  const key = provider.getKey(env);
  if (!key) {
    console.log(`[${provider.name}] no API key configured – skipping`);
    return null;
  }

  const payload = cloneBodyForProvider(body, provider);

  console.log(`[${provider.name}] attempting → model=${payload.model}`);

  const resp = await fetch(provider.url, {
    method: "POST",
    headers: {
      authorization: `Bearer ${key}`,
      "content-type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  if (resp.ok) {
    console.log(`[${provider.name}] OK (${resp.status})`);
    return resp;
  }

  const text = await resp.text().catch(() => "");
  console.log(
    `[${provider.name}] FAIL (${resp.status}): ${text.slice(0, 300)}`
  );

  // Rate limits and server errors → try next provider
  if (resp.status === 429 || resp.status >= 500) {
    return null;
  }

  // 4xx client error → propagate to caller (bad payload won't be fixed
  // by a different provider)
  throw new Response(text, {
    status: resp.status,
    headers: resp.headers,
  });
}

// =========================================================================
// Main handler
// =========================================================================

export default {
  async fetch(
    request: Request,
    env: Env,
    _ctx: ExecutionContext
  ): Promise<Response> {
    const url = new URL(request.url);

    if (
      request.method !== "POST" ||
      !url.pathname.endsWith("/chat/completions")
    ) {
      return json(
        { error: "POST /v1/chat/completions is the only supported endpoint" },
        404
      );
    }

    const apiType = (request.headers.get("X-API-Type") || "").toLowerCase();
    if (!["vision", "text"].includes(apiType)) {
      return json(
        { error: "X-API-Type header must be 'vision' or 'text'" },
        400
      );
    }

    let body: ChatCompletionBody;
    try {
      body = await request.json() as ChatCompletionBody;
    } catch {
      return json({ error: "Invalid JSON body" }, 400);
    }

    const chain = apiType === "vision" ? VISION_CHAIN : TEXT_CHAIN;

    // Each provider gets 15 seconds before we move on
    const PER_PROVIDER_MS = 15000;

    for (const provider of chain) {
      try {
        const ac = new AbortController();
        const timer = setTimeout(() => ac.abort(), PER_PROVIDER_MS);

        const resp = await tryProvider(provider, body, ac.signal, env);
        clearTimeout(timer);

        if (resp) {
          // Pipe the response body straight through so the client sees a
          // transparent OpenAI-compatible response.
          return new Response(resp.body, {
            status: resp.status,
            headers: {
              "content-type": "application/json",
              "x-proxy-provider": provider.name,
            },
          });
        }
      } catch (e) {
        if (e instanceof Response) throw e; // 4xx – propagate
        console.log(`[${provider.name}] exception: ${(e as Error).message}`);
        // fall through to next provider
      }
    }

    return json(
      { error: `All ${apiType} providers exhausted` },
      502
    );
  },
};
