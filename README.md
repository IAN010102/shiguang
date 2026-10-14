# Little Finds (拾光小本)

A pocket notebook that turns the things you screenshot into things you actually come back to.

Live demo: https://littlefinds.pages.dev/ · License: MIT

> Built for the Nebius x NVIDIA Global AI Hackathon, Best Apps & Agents track.

## The problem

We screenshot events, cafés, books and deadlines, feel like we've "saved" them, and never look again.

## What it does

Paste text or upload a screenshot. AI reads it, picks the right tab (quotes, cafés, events, recommendations, wishlist, books and more), pulls out the fields that matter (name, date, deadline, how to sign up, next step), and flags anything it is not sure about instead of guessing.

Then the app brings things back at the right time, and it treats two kinds of things differently:

- **Events and deadlines are reminders.** They show up under "Coming up soon" with a countdown and the deadline, and need no review.
- **Notes, quotes and lecture content are for review.** The AI writes a short summary and one recall question; the item returns on a spaced schedule (1, 3, 7, 14, 30, 60 days) and you grade it Forgot / Hazy / Got it.

## Agent behaviours

1. **Sorts and extracts.** One paste is enough; "Keep it" saves it into the right tab with the fields filled in.
2. **Splits.** A poster or list with several events becomes several cards you can tick off.
3. **Admits doubt.** Fields it cannot read or is unsure about are listed ("Unsure: year") and unreadable parts of a screenshot are marked, not invented.
4. **Checks the web.** For cafés, places and events it searches the web (Tavily), and compares what it finds with what you saved. If the page names the same shop and a different street address, it says so, links the source, and offers "Use the online info". The address comparison is done in code, not by the model, so a wrong address cannot be waved through. A claim is only shown when the page text actually contains it.
5. **Falls back safely.** If AI is down, slow (45 s), or rate-limited, the original rule-based sorter takes over.

Known limits: the online check can only confirm what a public web page says (chain stores may match a different branch, and a blog is not an official source); event dates are checked by the model and are less reliable than addresses.

## How AI is used

| Stage | Model / tool | Served by | Job |
|---|---|---|---|
| 1. Read | `zai-org/GLM-5.3-Flash` (vision) | Nebius Token Factory | Transcribe a screenshot, ignore phone UI, mark unclear text as `[uncertain]` |
| 2. Judge | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | Nebius Token Factory | Classify into a tab, extract fields, write a next step, notes, a summary and a one-question self-check |
| 3. Check | Tavily Search + Nemotron | Tavily / Nebius | Look the place or event up online and compare (a separate follow-up request, so the card appears first) |

- Nemotron is the core decision-maker. Text-only input skips stage 1.
- Sorting runs with the model's long chain of thought switched off for speed (set `AI_THINKING=1` to turn it back on).
- Code: [`functions/api/ai.js`](functions/api/ai.js). API keys stay server-side.

## Run it

The app is a static page plus one Cloudflare Pages Function.

```bash
npm i -g wrangler
printf "NEBIUS_API_KEY=your_key\nTAVILY_API_KEY=your_key\n" > .dev.vars   # never commit this
wrangler pages dev .
```

Deploy: connect the repo to Cloudflare Pages (no build step, output directory `/`) and add the secrets under Settings → Variables and Secrets.

| Variable | Required | Purpose |
|---|---|---|
| `NEBIUS_API_KEY` | yes | Nebius Token Factory (Nemotron, GLM) |
| `TAVILY_API_KEY` | no | Enables the online check; without it that step is skipped |
| `PER_IP_PER_HOUR` | no | Request cap per visitor (default 40) |
| `AI_THINKING` | no | `1` re-enables the model's long reasoning (slower) |

Without a key the app still works with rule-based sorting.

## Where NVIDIA Nemotron and Nebius Token Factory are used

- **NVIDIA Nemotron** (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) makes every sorting decision: which tab, which fields, the next step, the notes, the study summary and the recall question, and (for non-address claims) the online fact-check comparison.
- **Nebius Token Factory** serves both models through its OpenAI-compatible `chat/completions` endpoint, from a Cloudflare Pages Function at runtime (not at build time). The same endpoint serves the vision model that reads screenshots (`zai-org/GLM-5.3-Flash`).
- **Tavily** is called at runtime from the same function to search the web for cafés, places and events (see "Agent behaviours").

## Timeline

The project was started during the Submission Period (early September 2026). The rule-based version of the app was first deployed on October 9. Since October 13 the AI layer was added on top of it:

- The AI pipeline above, including screenshot input and field extraction with uncertainty flags
- Online verification with sources, and splitting one image into several event cards
- Reminder vs. review: a "Coming up soon" reminder list plus spaced-repetition review with AI-written recall questions
- A read-only event detail view with a concrete next step
- A one-tap "AI paste" entry point on every screen
- Server-side proxy with rate limiting and timeouts
- English UI pass; first visit follows the browser language

## Data and privacy

Notes are stored in your browser's localStorage. Only the text or image you explicitly send to "Sort with AI" leaves your device.
