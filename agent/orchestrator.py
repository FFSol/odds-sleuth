"""
Agent orchestrator — runs the OpenAI agentic tool loop.

Two modes:
  - generate_briefing(): runs the full analysis pipeline and returns a structured briefing
  - chat_response(): answers follow-up questions grounded in the briefing + tool access

Both support streaming via async generators that yield SSE-formatted chunks.
"""

from __future__ import annotations

import json
import os
from typing import AsyncGenerator

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from agent.tools import TOOL_SCHEMAS, dispatch_tool

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _get_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    return AsyncOpenAI(api_key=api_key)


MODEL = "gpt-4o"

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BRIEFING_SYSTEM_PROMPT = """You are an expert NBA sports betting analyst. You have access to a set of tools that query and analyze live odds data across 8 sportsbooks for tonight's 10 NBA games.

Your job is to generate a daily market briefing — the kind of report a sharp analyst reviews each morning to understand market health and spot opportunities.

## How to work

Use the tools systematically, in this order:
1. Call `get_all_games` to understand what's on the slate tonight
2. Call `detect_stale_lines` to find books with outdated prices
3. Call `detect_outlier_prices` to find suspicious or off-market odds
4. Call `detect_arbitrage_opportunities` to find guaranteed-profit situations
5. Call `rank_sportsbooks` to assess overall book quality
6. For any game that has anomalies or interesting lines, call `calculate_game_metrics` to get full vig/implied-prob analysis
7. Call `get_live_draftkings_odds` to fetch and publish the latest DraftKings lines for all games

Do NOT dump all data into your answer. Use tools to fetch only what you need, when you need it.

## Briefing format

Structure your briefing with these exact sections:

### 📊 Market Overview
Briefly describe tonight's slate: number of games, overall market health, any broad patterns.

### ⚠️ Anomalies Flagged
List every flagged issue with specifics:
- **Stale lines**: which book, which game, how many hours behind the market, and what risk that poses
- **Outlier prices**: which book, which game/market, what the odds are vs consensus, and the z-score

### 💰 Arbitrage Opportunities
List any cross-book arb opportunities with:
- The two legs (book, odds, side)
- Profit percentage
- Optimal stake split per $100

### 🎯 Top Value Opportunities
For each game, identify which book offers the best line on each side and by how much vs consensus. Highlight the top 3 most meaningful edges.

### 📈 Sportsbook Quality Rankings
Rank all 8 books with their composite score, avg vig %, and any notable issues.

### 🔍 Analysis Notes
Any patterns you noticed, caveats about the data, or things a bettor should know before acting on this briefing.

### 🏀 DraftKings Live Lines
Present the live DraftKings odds fetched via `get_live_draftkings_odds`. For each game list:
- Moneyline (home / away)
- Spread (line and price for each side)
- Total (over/under line and price)
Note the data source (live API or cached fallback) and the fetch timestamp.

## Rules
- Always show your math (vig %, implied probability, z-scores). Don't hand-wave numbers.
- If a stale line and an outlier flag point to the same book/game, connect those dots explicitly.
- Be direct and precise. This briefing is for professionals, not casual fans.
- If you're uncertain about something, say so — don't fabricate confidence.
"""

CHAT_SYSTEM_PROMPT = """You are an expert NBA sports betting analyst. You generated a daily market briefing that is included in this conversation.

You have access to the same tools used to generate that briefing, so you can look up specific data to answer follow-up questions accurately.

## Rules
- Ground your answers in the actual data — use tools when needed to fetch specifics
- Show your math when asked about calculations
- If a question is outside the scope of tonight's data, say so clearly
- Be concise and direct — analysts ask precise questions, give precise answers
- If you're uncertain, say so rather than guessing
"""

# ---------------------------------------------------------------------------
# Core agentic loop
# ---------------------------------------------------------------------------

async def _run_agent_loop(
    messages: list[ChatCompletionMessageParam],
    stream_text: bool = True,
) -> AsyncGenerator[str, None]:
    """
    Core agentic loop. Yields SSE-formatted strings:
      - data: {"type": "tool_call", "name": "...", "args": {...}}
      - data: {"type": "tool_result", "name": "...", "result": "..."}
      - data: {"type": "text_delta", "content": "..."}
      - data: {"type": "done"}
    """
    client = _get_client()

    while True:
        if stream_text:
            # Streaming mode: collect tool calls, stream text
            stream = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                stream=True,
            )

            # Accumulate the streamed response
            accumulated_tool_calls: dict[int, dict] = {}
            accumulated_text = ""
            finish_reason = None

            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                finish_reason = chunk.choices[0].finish_reason or finish_reason

                # Stream text content
                if delta.content:
                    accumulated_text += delta.content
                    yield f"data: {json.dumps({'type': 'text_delta', 'content': delta.content})}\n\n"

                # Accumulate tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in accumulated_tool_calls:
                            accumulated_tool_calls[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function else "",
                                "args_str": "",
                            }
                        if tc.id:
                            accumulated_tool_calls[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                accumulated_tool_calls[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                accumulated_tool_calls[idx]["args_str"] += tc.function.arguments

            # Process tool calls if any
            if accumulated_tool_calls:
                # Build assistant message with tool_calls
                tool_calls_msg = []
                for idx in sorted(accumulated_tool_calls.keys()):
                    tc = accumulated_tool_calls[idx]
                    tool_calls_msg.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["args_str"],
                        },
                    })

                messages.append({
                    "role": "assistant",
                    "content": accumulated_text or None,
                    "tool_calls": tool_calls_msg,
                })

                # Execute each tool and append results
                for tc in tool_calls_msg:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"]["arguments"])
                    except json.JSONDecodeError:
                        args = {}

                    yield f"data: {json.dumps({'type': 'tool_call', 'name': name, 'args': args})}\n\n"

                    result = dispatch_tool(name, args)

                    yield f"data: {json.dumps({'type': 'tool_result', 'name': name, 'result_preview': result[:300] + '...' if len(result) > 300 else result})}\n\n"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    })

                # Continue the loop
                continue

            # No tool calls — we're done
            break

        else:
            # Non-streaming fallback (not used in current UI but kept for tests)
            response = await client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)
                    result = dispatch_tool(name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                continue

            if msg.content:
                yield f"data: {json.dumps({'type': 'text_delta', 'content': msg.content})}\n\n"
            break

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_briefing() -> AsyncGenerator[str, None]:
    """Run the full briefing generation pipeline."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": BRIEFING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Generate tonight's market briefing. Run your full analysis pipeline: "
                "check all games, detect stale lines and outliers, find arbitrage, "
                "calculate metrics for games with anomalies, and rank the books. "
                "Then produce the structured briefing."
            ),
        },
    ]
    async for chunk in _run_agent_loop(messages, stream_text=True):
        yield chunk


async def chat_response(
    briefing: str,
    conversation_history: list[dict],
    user_message: str,
) -> AsyncGenerator[str, None]:
    """Answer a follow-up question grounded in the briefing."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
        {
            "role": "assistant",
            "content": f"Here is the market briefing I generated:\n\n{briefing}",
        },
    ]

    # Replay conversation history
    for turn in conversation_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": user_message})

    async for chunk in _run_agent_loop(messages, stream_text=True):
        yield chunk
