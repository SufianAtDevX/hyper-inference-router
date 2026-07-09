# Hyper-Inference Router

**AMD Developer Hackathon: ACT II — Track 3 (Unicorn Track)**
**Live demo:** [hyper-inference-router-1.onrender.com](https://hyper-inference-router-1.onrender.com)
**Bonus tracks targeted:** Best Use of Gemma via Fireworks · Best AMD-Hosted Gemma Project

---

## The problem

Every AI product that offers "unlimited" AI-generated content — proposals,
captions, reports, code explanations — is quietly making the same tradeoff
on every single request: send it to a big, expensive, general-purpose model
and eat the cost, or send it to a cheap model and risk a worse answer on
the requests that actually need reasoning depth.

Most products pick one model and use it for everything. That's simple, but
it means you're either overpaying for a "write a caption" request that
didn't need a frontier model, or underpowering a "write me a winning
proposal" request that needed real reasoning to be worth the user's time.

**Hyper-Inference Router solves this by routing, not picking.** It looks
at what a request actually needs and sends it to the right model for that
specific job — a fast, cheap model for simple requests, and **Gemma 4 26B
A4B IT** (Google DeepMind), running on a dedicated **AMD Instinct MI300X**
deployment via Fireworks AI, for the requests that need real reasoning or
creative quality.

## This isn't a hackathon toy — it's already running in production

This routing pattern was built for and is used inside
[**Kronos AI**](https://kronos.devxhouse.com), a live SaaS platform for
freelancers and agencies that generates job proposals, social media
content, and business documents on demand. Kronos's paying users generate
thousands of AI calls a month across wildly different request types — a
two-line social caption and a five-paragraph client proposal have
completely different quality and reasoning requirements, and billing them
identically doesn't make sense.

This repository is a **clean, standalone extraction** of that exact routing
layer, rebuilt without any of Kronos's business logic, user data, or
proprietary code — built specifically so judges and other developers can
see, run, and verify the routing pattern in isolation. What you're testing
here is the same architecture already deciding, in production, which model
answers a real paying customer's request.

## What it does

Every prompt is classified into one of four task types, each mapped to the
model actually suited for it:

| Task type | Model | Why |
|---|---|---|
| `casual` | DeepSeek V4 Flash (serverless, AMD MI300X) | Ultra-low latency and near-zero cost for greetings and simple queries — a large model here is wasted spend |
| `creative` | **Gemma 4 26B A4B IT** (dedicated, AMD MI300X) | MoE architecture gives strong creative-writing quality at a fraction of the active-parameter cost of a dense model of similar capability |
| `reasoning` | **Gemma 4 26B A4B IT** (dedicated, AMD MI300X) | Same model, used for structured analysis, proposals, and strategy — the class of request where quality actually matters to the end user |
| `code` | Kimi K2 Code (serverless, AMD MI300X) | Code-specialized model for technical docs and explanations |

Classification is a zero-cost weighted keyword match (see `classify_task()`
in `app.py`) — no extra LLM call is spent just deciding where to route,
which matters because a routing layer that itself costs money on every
request defeats its own purpose.

## Why Gemma 4 on a dedicated AMD MI300X deployment

Gemma 4 26B A4B IT is a Mixture-of-Experts model: 25.2B total parameters,
only ~3.8B active per forward pass. That architecture is exactly why it's
the right fit for the reasoning/creative tier — it delivers large-model
reasoning quality without large-model active-compute cost per token.

Running it on a **dedicated** Fireworks deployment (rather than shared
serverless) gives predictable latency and no per-request rate limits once
warm, at the honest tradeoff of GPU-hour billing while the deployment is
active and a cold-start delay after idle periods. The router's `/api/stats`
endpoint tracks real latency, token usage, and cost on every call — so that
tradeoff is measured live, not asserted in a slide.

## Live proof, not a claim

Every response returned by `/api/route` includes the actual model used, real
latency in milliseconds, real token counts, real cost in USD, and a live
comparison against equivalent GPT-4o pricing — computed from the same call,
not a canned number. Hit `/api/stats` on the running deployment at any time
to see the real running totals across every call made so far. Nothing in
this submission's numbers is simulated.

## Architecture

```
Client (browser demo UI or any HTTP client)
        |
        v
   FastAPI app (app.py)
        |
        v
   classify_task()  --- zero-cost weighted keyword classifier
        |
        v
   MODEL_REGISTRY lookup
        |
        v
   Fireworks AI Chat Completions API
        |
        +--> casual/code         -> serverless models (pay-per-token)
        +--> creative/reasoning  -> Gemma 4 26B A4B IT (dedicated AMD MI300X deployment)
```

## Running it

### 1. Get API access

- Sign up for the [AMD AI Developer Program](https://www.amd.com/en/developer/resources/ai-developer-program.html) to get $50 in Fireworks AI credits.
- In the [Fireworks dashboard](https://fireworks.ai), create an **on-demand deployment** of `Gemma 4 26B A4B IT` (Deployments -> New Deployment). Copy the resulting deployment ID — it looks like `accounts/<your-account>/deployments/<id>`.

### 2. Configure

```bash
cp .env.example .env
# Edit .env:
#   FIREWORKS_API_KEY=<your key>
#   GEMMA4_DEPLOYMENT_ID=accounts/<your-account>/deployments/<id>
```

### 3. Run with Docker (recommended)

```bash
docker compose up --build
```

Then open **http://localhost:8000** for the demo UI, or use the API directly (see below).

### 4. Run locally without Docker

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API

**`POST /api/route`**

```bash
curl -X POST http://localhost:8000/api/route \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a proposal for a Python developer role on a healthcare platform.",
    "task_hint": "reasoning"
  }'
```

Response:
```json
{
  "text": "...",
  "model": "accounts/.../deployments/...",
  "model_display": "Gemma 4 26B A4B IT (AMD MI300X, dedicated)",
  "task_type": "reasoning",
  "latency_ms": 1830,
  "tokens_used": 214,
  "cost_usd": 0.0000963,
  "gpt4_equivalent_cost_usd": 0.00107,
  "savings_vs_gpt4_pct": 91.0
}
```

`task_hint` is optional — omit it to let the classifier decide. Valid
values: `casual`, `creative`, `reasoning`, `code`.

**`GET /api/stats`** — running totals and the last 10 calls (cost, latency, model used) across the live deployment.

**`GET /api/health`** — readiness check, confirms API key and deployment ID are configured.

## Notes on cost

Dedicated Fireworks deployments bill by GPU-hour while warm, separately
from per-token cost, and can take one to several minutes to cold-start
from zero replicas. For local testing, expect the first request after
idle time to be slow (or to time out and need a retry) while the
deployment wakes up. Scale the deployment to 0 replicas (or delete it)
in the Fireworks dashboard when not actively demoing to avoid idle billing.

## What's next

The routing pattern here is intentionally minimal so it's easy to verify —
a real product deployment (as in Kronos AI) layers on top of it: per-user
usage tracking, a fallback chain if a model provider errors out, and
richer task classification. Those layers were deliberately left out of
this submission to keep the AMD/Gemma 4 routing logic itself fully
visible and auditable, rather than buried in unrelated application code.

## License

MIT — see [LICENSE](LICENSE).
