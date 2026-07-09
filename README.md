# Hyper-Inference Router

**AMD Developer Hackathon: ACT II — Track 3 (Unicorn Track) submission**

A cost- and latency-aware task router that classifies incoming requests and
sends each one to the cheapest Fireworks AI model capable of handling it
well — with **Gemma 4 26B A4B IT** (Google DeepMind), running on a
dedicated **AMD Instinct MI300X** deployment via Fireworks AI, as the engine
for reasoning and creative tasks.

This project was extracted from a production feature inside
[Kronos AI](https://kronos.devxhouse.com), a freelancer assistant platform,
where this exact routing logic powers real proposal generation and content
creation for paying users. This repo is a clean, standalone version built
specifically to demonstrate that routing layer for the hackathon — it does
not include the rest of the Kronos AI application or its business logic.

## What it does

Every prompt is classified into one of four task types, each mapped to a
different model based on what it actually needs:

| Task type | Model | Why |
|---|---|---|
| `casual` | DeepSeek V4 Flash (serverless, AMD MI300X) | Ultra-low latency for greetings/simple queries — no need for a large model |
| `creative` | **Gemma 4 26B A4B IT** (dedicated, AMD MI300X) | MoE architecture gives strong creative writing quality at a fraction of the active-parameter cost of a dense model |
| `reasoning` | **Gemma 4 26B A4B IT** (dedicated, AMD MI300X) | Same model, used for structured analysis, proposals, and strategy — Gemma 4's reasoning mode handles both well |
| `code` | Kimi K2 Code (serverless, AMD MI300X) | Code-specialized model for technical docs and explanations |

Classification is a zero-cost keyword match (see `classify_task()` in
`app.py`) — no extra LLM call is spent just to decide where to route.

## Why Gemma 4 on a dedicated AMD MI300X deployment

Gemma 4 26B A4B IT is a Mixture-of-Experts model: 25.2B total parameters,
only ~3.8B active per forward pass. Running it on a dedicated Fireworks
deployment (rather than a shared serverless endpoint) gives predictable
latency and no per-request rate limits, at the tradeoff of GPU-hour billing
while the deployment is warm. The router's `/api/stats` endpoint tracks
real latency and cost per call so that tradeoff is measurable, not assumed.

## Architecture

```
Client (browser demo UI or any HTTP client)
        |
        v
   FastAPI app (app.py)
        |
        v
   classify_task()  --- zero-cost keyword classifier
        |
        v
   MODEL_REGISTRY lookup
        |
        v
   Fireworks AI Chat Completions API
        |
        +--> casual/code  -> serverless models (pay-per-token)
        +--> creative/reasoning -> Gemma 4 26B A4B IT (dedicated AMD MI300X deployment)
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
    "prompt": "Write a proposal for a Python developer role on a healthcare API platform.",
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
  "cost_usd": 0.0000963
}
```

**`GET /api/stats`** — running totals and the last 10 calls (cost, latency, model used).

**`GET /api/health`** — readiness check, confirms API key and deployment ID are configured.

## Notes on cost

Dedicated Fireworks deployments bill by GPU-hour while warm, separately
from per-token cost, and can take one to several minutes to cold-start
from zero replicas. For local testing, expect the first request after
idle time to be slow (or to time out and need a retry) while the
deployment wakes up. Scale the deployment to 0 replicas (or delete it)
in the Fireworks dashboard when not actively demoing to avoid idle billing.

## License

MIT — see [LICENSE](LICENSE).
