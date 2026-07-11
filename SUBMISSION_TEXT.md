# lablab.ai Submission Text — Hyper-Inference Router

Copy/paste these directly into the submission form fields. Not committed
to the public repo intentionally (it's a working doc, not app code) —
keep it locally or paste into the form now.

---

## Project Title

Hyper-Inference Router: Cost-Aware AI Routing on AMD MI300X

(Alternative, shorter: "Hyper-Inference Router")

---

## Short Description (1-2 sentences)

A routing layer that sends every AI request to the cheapest model that
can actually handle it well — using Gemma 4 E4B on a dedicated AMD
Instinct MI300X deployment for reasoning and creative tasks. Already
running in production inside Kronos AI, a live SaaS platform.

---

## Long Description

**(Verified 1875 characters — under lablab.ai's 2000 char limit)**

Most AI products pick one model and use it for every request, regardless of what that request needs. That means overpaying a frontier model for a two-word greeting, or underpowering a request that needed real reasoning.

Hyper-Inference Router fixes this by classifying every prompt and routing it to the model suited to the job:

- Casual/simple queries -> a fast, cheap serverless model
- Reasoning and creative tasks (proposals, strategy, marketing copy) -> Gemma 4 E4B (Google DeepMind), on a dedicated AMD Instinct MI300X deployment via Fireworks AI
- Code-related queries -> a code-specialized model

Classification is a zero-cost weighted keyword match, so routing adds no LLM-call overhead of its own.

This isn't a hackathon-only prototype. The routing pattern was built for and is used inside Kronos AI (kronos.devxhouse.com), a live SaaS platform for freelancers and agencies generating job proposals, social content, and business documents for paying users. This submission is a clean, standalone extraction of that exact logic, rebuilt without Kronos's business code, so it can be run and verified in isolation.

Every response returns real numbers: model used, latency, tokens, cost in USD, and a live comparison against GPT-4o pricing, computed from the same call. Gemma 4 E4B (8B total / ~4.5B effective) is cheap enough to run dedicated while giving a real quality step up over cheap serverless models on tasks needing actual reasoning.

Who pays and why: any team running AI at scale wastes money without task-aware routing. This is the routing layer that plugs into any Fireworks-based product, already proven inside a paying SaaS product, cutting spend on the majority of requests that never needed a frontier model.

Try it live: submit any prompt at the demo URL and watch it route in real time, with the actual model, cost, and latency shown on screen.

---

## Technology Tags

AMD MI300X, Fireworks AI, Gemma 4, Google DeepMind, FastAPI, Docker,
LLM Routing, Cost Optimization, Python

## Category Tags

AI Infrastructure / Developer Tools, Cost Optimization, LLM Orchestration

---

## Cover Image

Use a screenshot of the live demo UI showing a completed request with the
model badge ("Gemma 4 E4B (AMD MI300X, dedicated)") and the cost/
latency/savings line visible — that's the single most convincing visual,
since it proves the routing and the cost claim in one frame.

## Application URL

https://hyper-inference-router-1.onrender.com

## GitHub Repository

https://github.com/SufianAtDevX/hyper-inference-router
