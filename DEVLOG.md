# Development Log

## How I used CC during development

This project was built using Claude Code (CC) as the primary coding agent (VS Code IDE) and within I'll account what I delegated, I wrote myself, and general changes/considerations. Disclaimer: I do not write Devlogs routinely so this doc has been cleaned up by me after a rewrite by CC. 

### What I delegated to AI

**Project scaffolding and architecture** — I described the requirements (FastAPI + OpenAI + SSE streaming + no separate frontend) and had Claude scaffold the full directory structure, pyproject.toml, and file layout in one pass. This saved the first 30 minutes of setup.

**Boilerplate code** — The FastAPI routes (`main.py`), Pydantic models, and SSE response patterns are patterns I know well but are tedious to type. These were generated and reviewed rather than written from scratch.

**The HTML/JS frontend** — The entire `static/index.html` was generated from a description of the UI needs: left panel for briefing, right panel for chat, SSE streaming, tool call trace display, minimal dark theme. I reviewed and adjusted the streaming event handling logic.

**OpenAI tool schemas** — The JSON schema definitions in `TOOL_SCHEMAS` were generated from descriptions of what each tool does. These are verbose and error-prone to write manually.

### What I wrote (or significantly modified) myself

**`agent/math.py`** — The odds math formulas were mostly written by me and verified by hand. LLMs tend to hallucinate absolute math (not sure why) so I didn't trust it. I ran the funcs in a regular Python repl, nothing crazy.
- American → implied probability conversion (verified: -150 → 60.0%, +200 → 33.33%, -110 → 52.38%)
- Vig calculation (verified: -110/-110 → 4.76% vig)
- Arbitrage detection (verified: DEN@MIL best lines sum to 92.49% → 7.51% profit)
- Staleness detection (relative, not absolute — this was a key design decision)
- Outlier detection (z-score in implied-probability space, not raw odds space)

**Agent system prompts** — The system prompts in `orchestrator.py` went through multiple iterations (see below). The final versions were rewritten by me with  assistance for formatting.

**Tool implementations** — The logic in `detect_stale_lines`, `detect_outlier_prices`, and `detect_arbitrage_opportunities` needed a bit of work for verification since I lacked the domain knowledge so CC generated the structure and I reviewed then partially rewrote the core detetc logic.

---

## Prompt iterations

### Briefing system prompt evolution (summarized)

**v1:**
The agent called tools but then generated a narrative with no consistent structure. Follow-up questions were hard to ground because the briefing lacked clear sections.

**v2 (prescribed tool call order + output structure):**
Added explicit step by step instructions for which tools to call in what order, and defined the exact briefing sections with emoji headers. Also added: "Always show your math (vig %, implied probability, z-scores). Don't hand-wave numbers."

The call and prompt structure matters a lot to CC because it tends to expand context repeatedly until an answer is found. This change had the data become consistent, the calcs actually showed up, and stale lines/outliers/arb opportunities were confirmed based on the samples provided.

**v3 (current — added domain rules):**
Since I'm not very knowledgeable on money lines and how they look analytically, I did light research in a separate session with CC and asked my v2 session to add domain knowledege level rules to further improve things. i.e. "if a stale line and an outlier flag point to the same book/game, connect those dots explicitly." 

### Chat system prompt issues

**v (current):** Specifing consistently to "use tools when needed to fetch specifics" and "base answers in actual data" keep returning stale results even if I modify the data, which makes me believe the Agent can't determine what needs to be recomputed vs implicitly (?) cached. Currently the agent calls `calculate_game_metrics` or `get_game_odds` when asked for specific numbers rather than recalling them from the briefing but I can't get it to remain consistent. Hopefully this is the only inconsistency (bug, I suppose..).

---

## Infra/Architecture choices

### FastAPI + SSE
SSE is simpler for unidirectional server→client streaming and works over standard HTTP. Other frameworks add complexity (connection management, reconnection logic) that would require me having more domain knowledge there, so I stuck to what I know. SSE also works naturally with `fetch()` + `ReadableStream` in the browser.

### Why one service instead of separate frontend?
FastAPI serves the static HTML directly. This means one process to deploy, one URL, no CORS configuration, and no frontend build step. For a tool this size, the added complexity of a separate React app isn't justified.

### Relative staleness decision (possibly linked to Chat system prompt issue)
`datetime.now()` as the staleness baseline means the threshold would need to change depending on when the agent runs. Relative comparison (book A's timestamp vs. the freshest book for the same game) is more robust. But this may also be why certain stale answers are being returned because of some time logic that isn't being properly caught.

---

### CC Issues

1. **Staleness detection (first attempt)** — The initial implementation compared `last_updated` against wall clock time (`datetime.now()`). This is wrong for a static dataset where all timestamps are in the past — everything would appear stale. Fixed by comparing each book's timestamp against the **maximum timestamp for that game** (the freshest book), not the current time.

2. **Outlier detection (first attempt)** — Initial version computed z-scores on raw American odds values. This produces misleading results because of the discontinuity around ±100. I didn't actually catch this initially, but CC did and converted to implied probability first, then compute z-scores.
