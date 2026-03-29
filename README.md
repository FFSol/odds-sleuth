# odds-sleuth

An AI-powered NBA odds agent that detects anomalies, analyzes market value, and generates daily briefings a human analyst can act on.

## What it does

- **Detect** — Flags stale lines (books significantly behind the market), statistical outlier prices (z-score > 2σ), and potential data errors
- **Analyze** — Calculates implied probability, vig/margin, no-vig fair odds, and best available line for each market
- **Brief** — Generates a structured daily market briefing with anomalies, arbitrage opportunities, value plays, and sportsbook quality rankings
- **Chat** — Supports follow-up questions grounded in the briefing and backed by live tool calls

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11 + FastAPI |
| AI | OpenAI gpt-4o with function calling |
| Frontend | Single-page HTML/JS (no framework) |
| Streaming | Server-Sent Events (SSE) |
| Deployment | Railway |

## Architecture

```
User
  │
  ├─ GET /           → static/index.html
  ├─ POST /api/brief → SSE stream (agent generates briefing)
  └─ POST /api/chat  → SSE stream (agent answers follow-up)
       │
       ▼
  agent/orchestrator.py   (OpenAI agentic loop)
       │
       ├─ agent/tools.py  (7 tool implementations + OpenAI schemas)
       │       │
       │       └─ agent/math.py  (implied prob, vig, arb, outlier math)
       │
       └─ data/odds.json  (10 NBA games × 8 sportsbooks)
```

### Agent tool loop

The agent uses OpenAI function calling in a `while True` loop: it calls tools, receives results, decides what to call next, and continues until it's ready to write the final briefing. The LLM never sees the raw JSON dump of all 80 records — it queries specific tools for specific data.

### Tools

| Tool | Purpose |
|---|---|
| `get_all_games` | List all games and available books |
| `get_game_odds` | Raw odds for a specific game |
| `calculate_game_metrics` | Vig, implied prob, fair odds, best lines |
| `detect_stale_lines` | Books lagging behind the market (relative timestamps) |
| `detect_outlier_prices` | Z-score analysis across all markets |
| `detect_arbitrage_opportunities` | Cross-book arb with stake split calculator |
| `rank_sportsbooks` | Composite quality score: vig + freshness + accuracy |

### Odds math

All calculations are deterministic Python in `agent/math.py` — the LLM never approximates numbers:

- **Implied probability**: negative odds → `|odds| / (|odds| + 100)`, positive → `100 / (odds + 100)`
- **Vig**: `sum(implied probs) - 1`
- **Fair odds**: normalize implied probs to sum to 100%
- **Arbitrage**: `best_implied_prob(A) + best_implied_prob(B) < 1.0`
- **Staleness**: relative — compared against the freshest book for the same game, not wall clock

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- OpenAI API key

### Local development

```bash
# Clone and install
git clone <repo>
cd odds-sleuth
uv sync

# Set your API key
cp .env.example .env
# Edit .env and add: OPENAI_API_KEY=sk-...

# Run
uv run uvicorn main:app --reload
# Open http://localhost:8000
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | Your OpenAI API key |

## Deployment (Railway)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
railway domain   # set a public domain
```

The `railway.toml` in the repo handles process configuration automatically.

## Dataset

`data/odds.json` — 10 NBA games × 8 sportsbooks (80 records) with intentional anomalies:
- **Stale lines**: PointsBet on LAL@BOS (~9h behind), BetRivers on DAL@PHX (~6.8h behind), Caesars on ATL@CHA (~10.3h behind)
- **Outlier prices**: BetMGM moneyline on DEN@MIL (z-score 2.4σ)
- **Arbitrage**: DEN@MIL (7.5% profit), MIN@SAC (1.3%), LAL@BOS (0.7%), POR@UTA (0.3%)

## Running tests

```bash
uv run pytest
```