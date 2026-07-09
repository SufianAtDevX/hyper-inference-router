"""
Hyper-Inference Router — AMD Developer Hackathon: ACT II submission
=====================================================================
A cost/latency-aware task router that sends each request to the
cheapest Fireworks AI model capable of handling it well, with Gemma 4
26B A4B IT (Google DeepMind) as the dedicated reasoning/creative engine
running on AMD Instinct MI300X hardware.

Task taxonomy:
  casual     -> fast/cheap model   (greetings, short replies, classification)
  creative   -> Gemma 4 (dedicated)-> marketing copy, storytelling, captions
  reasoning  -> Gemma 4 (dedicated)-> proposals, strategy, business analysis
  code       -> code-specialized model -> technical docs, code explanations

Run:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in your FIREWORKS_API_KEY
    uvicorn app:app --host 0.0.0.0 --port 8000

Then open http://localhost:8000 for the demo UI, or POST to /route directly.
"""

import os
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("hyper_inference_router")

# ---------------------------------------------------------------------------
# Fireworks AI configuration — AMD Instinct MI300X backed
# ---------------------------------------------------------------------------
FW_BASE = "https://api.fireworks.ai/inference/v1"
FW_KEY = os.getenv("FIREWORKS_API_KEY", "")

# Dedicated Gemma 4 deployment on AMD MI300X (set after creating your own
# on-demand deployment in the Fireworks dashboard — see README).
GEMMA4_DEPLOYMENT_ID = os.getenv("GEMMA4_DEPLOYMENT_ID", "")

MODEL_REGISTRY = {
    "casual": {
        "id": "accounts/fireworks/models/deepseek-v4-flash",
        "display": "DeepSeek V4 Flash (AMD MI300X, serverless)",
        "max_tokens": 512,
        "temperature": 0.7,
        "cost_per_1k": 0.0001,
        "reason": "Ultra-low latency for greetings and simple classification tasks.",
    },
    "creative": {
        "id": GEMMA4_DEPLOYMENT_ID,
        "display": "Gemma 4 26B A4B IT (AMD MI300X, dedicated)",
        "max_tokens": 2048,
        "temperature": 0.85,
        "cost_per_1k": 0.00045,
        "reason": "Google DeepMind's Gemma 4 MoE model for creative writing, marketing copy, storytelling.",
    },
    "reasoning": {
        "id": GEMMA4_DEPLOYMENT_ID,
        "display": "Gemma 4 26B A4B IT (AMD MI300X, dedicated)",
        "max_tokens": 4096,
        "temperature": 0.5,
        "cost_per_1k": 0.00045,
        "reason": "Gemma 4's MoE reasoning path for structured analysis, proposals, and strategy.",
    },
    "code": {
        "id": "accounts/fireworks/models/kimi-k2p7-code",
        "display": "Kimi K2 Code (AMD MI300X, serverless)",
        "max_tokens": 2048,
        "temperature": 0.3,
        "cost_per_1k": 0.0009,
        "reason": "Code-specialized model for technical documentation and explanations.",
    },
}

# Keywords are weighted: strong, unambiguous signals score higher than
# generic terms that appear across multiple contexts (e.g. "api" alone is
# too weak a signal for "code" -- it shows up constantly in business/
# proposal prompts like "healthcare API project" that are really
# reasoning tasks, not programming tasks).
TASK_KEYWORDS = {
    "casual": {
        "hello": 2, "hi": 2, "hey": 2, "thanks": 2, "thank you": 2,
        "goodbye": 2, "bye": 2, "how are you": 2,
    },
    "code": {
        "function": 2, "debug": 2, "script": 2, "algorithm": 2,
        "python": 2, "javascript": 2, "code snippet": 3, "stack trace": 3,
        "refactor": 2, "unit test": 2, "endpoint": 1, "api": 1,
    },
    "reasoning": {
        "proposal": 3, "business plan": 3, "strategy": 2, "analyze": 2,
        "evaluate": 2, "recommend": 2, "compare": 2, "decision": 2,
        "plan": 1, "roadmap": 2, "report": 2, "budget": 2, "bidding": 2,
    },
    "creative": {
        "caption": 3, "post": 1, "story": 2, "blog": 2, "marketing": 3,
        "campaign": 2, "copy": 1, "creative": 2,
    },
}


def classify_task(prompt: str, hint: Optional[str] = None) -> str:
    """Classify a prompt into casual | creative | reasoning | code via
    weighted keyword match. Strong, unambiguous keywords (e.g. "proposal",
    "business plan") outweigh generic ones (e.g. "api") so common business
    language doesn't get misrouted to the code path."""
    if hint and hint in MODEL_REGISTRY:
        return hint
    lower = prompt.lower()[:500]
    scores = {task: 0 for task in TASK_KEYWORDS}
    for task, keywords in TASK_KEYWORDS.items():
        for kw, weight in keywords.items():
            if kw in lower:
                scores[task] += weight
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "reasoning"


def call_fireworks(prompt: str, task_type: str, system_prompt: str = "") -> dict:
    """Call Fireworks AI chat completions for the given task type."""
    if not FW_KEY:
        return {"text": None, "error": "FIREWORKS_API_KEY not set", "model": None}

    cfg = MODEL_REGISTRY.get(task_type, MODEL_REGISTRY["creative"])
    model_id = cfg["id"]

    if not model_id:
        return {"text": None, "error": f"No model configured for task '{task_type}' — "
                                        f"set GEMMA4_DEPLOYMENT_ID in .env", "model": None}

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": cfg["max_tokens"],
        "temperature": cfg["temperature"],
        "top_p": 0.9,
    }

    t0 = time.time()
    try:
        resp = requests.post(
            f"{FW_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {FW_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        latency_ms = int((time.time() - t0) * 1000)

        if resp.status_code != 200:
            logger.warning("Fireworks HTTP %s: %s", resp.status_code, resp.text[:200])
            return {"text": None, "error": f"HTTP {resp.status_code}: {resp.text[:150]}",
                    "model": model_id, "latency_ms": latency_ms}

        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens", 0)
        cost_usd = round(tokens_used * cfg["cost_per_1k"] / 1000, 6)

        # GPT-4o flat pricing for comparison ($5/1M input+output blended est.)
        gpt4_equivalent_cost = round(tokens_used * 0.005 / 1000, 6)
        savings_pct = (
            round((1 - cost_usd / gpt4_equivalent_cost) * 100, 1)
            if gpt4_equivalent_cost > 0 else 0
        )

        return {
            "text": text,
            "model": model_id,
            "model_display": cfg["display"],
            "task_type": task_type,
            "reason": cfg["reason"],
            "latency_ms": latency_ms,
            "tokens_used": tokens_used,
            "cost_usd": cost_usd,
            "gpt4_equivalent_cost_usd": gpt4_equivalent_cost,
            "savings_vs_gpt4_pct": savings_pct,
            "error": None,
        }
    except requests.exceptions.Timeout:
        return {"text": None, "error": "Fireworks request timed out (deployment may be cold-starting)",
                "model": model_id, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as exc:
        logger.exception("Fireworks call failed")
        return {"text": None, "error": str(exc), "model": model_id,
                "latency_ms": int((time.time() - t0) * 1000)}


# ---------------------------------------------------------------------------
# In-memory call log (for the demo dashboard — no database needed)
# ---------------------------------------------------------------------------
CALL_LOG: list[dict] = []


class RouteRequest(BaseModel):
    prompt: str
    task_hint: Optional[str] = None
    system_prompt: Optional[str] = ""


app = FastAPI(
    title="Hyper-Inference Router",
    description="Cost-aware task routing across AMD MI300X-hosted models via Fireworks AI, "
                "featuring Gemma 4 26B A4B IT for reasoning and creative tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "fireworks_key_configured": bool(FW_KEY),
        "gemma4_deployment_configured": bool(GEMMA4_DEPLOYMENT_ID),
        "time": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/route")
def route(req: RouteRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    task_type = classify_task(req.prompt, hint=req.task_hint)
    result = call_fireworks(req.prompt, task_type, system_prompt=req.system_prompt or "")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_preview": req.prompt[:80],
        "task_type": task_type,
        **{k: v for k, v in result.items() if k != "text"},
    }
    CALL_LOG.append(entry)

    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["error"])

    return result


@app.get("/api/stats")
def stats():
    if not CALL_LOG:
        return {"total_calls": 0, "total_cost_usd": 0, "avg_latency_ms": 0, "recent": []}

    total_calls = len(CALL_LOG)
    total_cost = sum(c.get("cost_usd", 0) or 0 for c in CALL_LOG)
    avg_latency = sum(c.get("latency_ms", 0) or 0 for c in CALL_LOG) / total_calls

    return {
        "total_calls": total_calls,
        "total_cost_usd": round(total_cost, 6),
        "avg_latency_ms": int(avg_latency),
        "recent": list(reversed(CALL_LOG[-10:])),
    }


# Serve the simple demo UI
if os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")
else:
    @app.get("/", response_class=HTMLResponse)
    def root():
        return "<h1>Hyper-Inference Router</h1><p>API is running. See /docs for the API reference.</p>"
