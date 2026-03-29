# DEVLOG

## How I used AI

Primary tool: Claude Code (claude-sonnet-4-6). Used it as a force multiplier, not a replacement.

**What I handed off:**
- Project scaffolding. Described the stack (FastAPI + OpenAI + SSE, single service, no separate frontend) and had it generate the directory layout, pyproject.toml, file stubs. Saved the first 30 minutes.
- FastAPI boilerplate. Routes, Pydantic models, SSE response setup. Know this stuff cold, just tedious to type.
- The frontend. Described the UI: briefing panel left, chat right, tool trace at the bottom, dark theme, SSE streaming. Got a working first pass. Adjusted the streaming event loop myself.
- OpenAI tool schemas. Verbose JSON schema definitions. Error-prone to write manually. Generated from descriptions, reviewed each one.

**What I wrote or rewrote myself:**
- `agent/math.py`. Odds math is the correctness-critical layer — I didn't trust generated code here. Wrote the formulas, verified by hand.
  - `-150` → 60.0% ✓, `+200` → 33.33% ✓, `-110/-110` → 4.76% vig ✓
  - DEN@MIL arb: best lines sum to 92.49% → 7.51% profit ✓
- Core detection logic in the tools. AI scaffolded the structure. I rewrote `detect_stale_lines`, `detect_outlier_prices`, and `detect_arbitrage_opportunities`.
- System prompts. Multiple iterations, written by me (details below).

**Where AI got it wrong:**

Staleness detection:
- First version compared `last_updated` against `datetime.now()`.
- Wrong. Static dataset — everything looked stale.
- Fixed: compare each book's timestamp against the max timestamp for that game. Relative, not absolute.

Outlier detection:
- First version ran z-scores on raw American odds.
- That's misleading. American odds are nonlinear — the gap from -110 to -120 isn't the same as +110 to +120 in probability terms.
- Fixed: convert to implied probability first, then compute z-scores.

SSE streaming:
- First version buffered the full response before yielding.
- Defeated the whole point.
- Fixed: yield text deltas as chunks arrive.

---

## Prompt iterations

### Briefing prompt

**v1:**
> "You are a sports betting analyst. Analyze the odds data and generate a report."

Result: dumped raw data, no tool calls, numbers were off. Useless.

**v2:** Added "use the tools, don't dump raw data."

Result: called tools but wrote unstructured prose. Follow-up questions were hard to ground. No consistent sections.

**v3:** Prescribed tool call order + exact output sections.

Added: "Always show your math. Don't hand-wave numbers."

Result: structured, consistent. Math showed up explicitly. Anomalies in predictable places.

**v4 (current):** Added "if a stale line and an outlier point to the same book/game, connect those dots."

This was the meaningful change. The agent now surfaces that PointsBet's stale LAL@BOS line *is* why their home ML is an outlier — they haven't moved with the market. That's the insight that's actually useful.

### Chat prompt

**v1:** "You're an analyst, answer questions."

Problem: answered from memory of the briefing instead of re-querying tools. Made up specific numbers.

**v2 (current):** Added "use tools when needed, ground answers in actual data."

Now it calls `calculate_game_metrics` or `get_game_odds` when asked for specifics instead of hallucinating.

---

## Key decisions

**FastAPI + SSE instead of WebSockets.**
Only needed server → client streaming. WebSockets add reconnection logic and state management for no reason here. SSE works fine with `fetch()` + `ReadableStream`.

**One service, no separate frontend.**
FastAPI serves the HTML directly. One process, one URL, no CORS, no build step. A React app would've been overkill.

**Relative staleness, not absolute.**
`datetime.now()` as the baseline breaks on static datasets and drifts on live ones. Comparing against the freshest book per game is more robust — catches books behind the market regardless of when you run it.

**Implied-probability space for outlier detection.**
Raw American odds are nonlinear. Z-scores on raw values don't mean anything useful. Convert to probability first.

**gpt-4o over gpt-4o-mini.**
Tried mini. It skipped tool calls and missed the connection between stale lines and outlier prices. 4o follows the analysis pipeline reliably. Worth the cost difference here.

**Tool trace in the UI, not raw results.**
Tool results are noisy (80 records of JSON). Showing the trace — what was called, with what args, and a short preview — is enough for transparency without overwhelming the user.

---

## What I'd fix with more time

**Live odds.** The static dataset works for evaluation. Real version polls The Odds API and updates incrementally. Tool layer is already set up for it — just swap `_load_data()` with an API client.

**Confidence scores on flags.** Right now everything is binary. A score based on z-score magnitude or lag hours would help analysts prioritize. Math layer already has the numbers. Presentation problem.

**Spread and total arb.** Only doing moneyline arb right now. Spread/total arb requires line normalization (e.g. `-5.5 at -110` vs `-6 at +100`). Skipped for time.

**Briefing history.** Each session starts fresh. SQLite store of past briefings would let you compare today vs yesterday. Probably a few hours of work.

**Better outlier context.** The prompt asks the agent to connect outlier + stale flags. It does this sometimes, not always. Inconsistent. Needs tighter prompt or a dedicated "explain this flag" tool.

**Tests for `agent/math.py`.** Nothing there yet. This is the one place where a bug actually matters. Property-based tests with `hypothesis` would be the right call — especially for edge cases like odds of exactly ±100.

**Rate limiting.** Nothing stops someone from hammering `/api/brief`. Easy to rack up an OpenAI bill. Token bucket on the endpoints would fix it.
