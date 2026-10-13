# Little Finds (拾光小本)

A pocket notebook that turns the things you screenshot into things you actually come back to.

Live demo: https://littlefinds.pages.dev/ · License: MIT

> Built for the Nebius x NVIDIA Global AI Hackathon, Best Apps & Agents track.

## The problem

We screenshot events, cafés, books and deadlines, feel like we've "saved" them, and never look again.

## What it does

Paste text or upload a screenshot. AI reads it, picks the right tab (quotes, cafés, events, recommendations, wishlist, books and more), pulls out the fields that matter (name, date, deadline, how to sign up, next step), and flags anything it is not sure about instead of guessing. The app then keeps surfacing a few items each day so they get reviewed, not buried.

## How AI is used

| Stage | Model | Served by | Job |
|---|---|---|---|
| 1. Read | `zai-org/GLM-5.3-Flash` (vision) | Nebius Token Factory | Transcribe a screenshot, mark unclear text as `[uncertain]` |
| 2. Judge | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | Nebius Token Factory | Classify into a tab, extract fields, write a next step and a one-question self-check |

- The Nemotron stage is the core decision-maker; text-only input skips stage 1.
- The original rule-based "smart paste" (`classify()` in `index.html`) is kept as an automatic fallback when AI is unavailable or rate-limited.
- Code: [`functions/api/ai.js`](functions/api/ai.js). The API key stays server-side.

## Run it

The app is a static page plus one Cloudflare Pages Function.

```bash
npm i -g wrangler
echo "NEBIUS_API_KEY=your_key" > .dev.vars     # never commit this
wrangler pages dev .
```

Deploy: connect the repo to Cloudflare Pages (no build step, output directory `/`) and add `NEBIUS_API_KEY` under Settings → Variables and Secrets. Optional: `PER_IP_PER_HOUR` (default 20).

Without a key the app still works with rule-based sorting.

## Updates during the Submission Period

The app existed before the hackathon (rule-based keyword sorting, local storage only). New during the period: the AI pipeline above, image input, field extraction with uncertainty flags, the server-side proxy with rate limiting, and an English UI pass.

## Data and privacy

Notes are stored in your browser's localStorage. Only the text or image you explicitly send to "Sort with AI" leaves your device.
