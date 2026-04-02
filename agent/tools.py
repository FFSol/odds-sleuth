"""
Tool implementations for the odds agent.

Each function is the actual Python implementation. TOOL_SCHEMAS below defines
the OpenAI function-calling schemas so the LLM knows what to call and when.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.math import (
    american_to_implied_prob,
    calculate_vig,
    calculate_fair_odds,
    best_line_for_side,
    detect_arbitrage,
    detect_outliers,
    minutes_since_update,
    score_book,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DATA_PATH = Path(__file__).parent.parent / "data" / "odds.json"
_cache: dict | None = None


def _load_data() -> dict:
    global _cache
    if _cache is None:
        with open(_DATA_PATH) as f:
            _cache = json.load(f)
    return _cache


def _records_by_game() -> dict[str, list[dict]]:
    data = _load_data()
    result: dict[str, list[dict]] = {}
    for record in data["odds"]:
        gid = record["game_id"]
        result.setdefault(gid, []).append(record)
    return result


# ---------------------------------------------------------------------------
# Tool 1: get_all_games
# ---------------------------------------------------------------------------

def get_all_games() -> dict:
    """Return a summary of all games in the dataset."""
    by_game = _records_by_game()
    games = []
    for gid, records in by_game.items():
        first = records[0]
        games.append({
            "game_id": gid,
            "home_team": first["home_team"],
            "away_team": first["away_team"],
            "commence_time": first["commence_time"],
            "books_available": [r["sportsbook"] for r in records],
        })
    return {"games": games, "total": len(games)}


# ---------------------------------------------------------------------------
# Tool 2: get_game_odds
# ---------------------------------------------------------------------------

def get_game_odds(game_id: str) -> dict:
    """Return raw odds for a specific game across all sportsbooks."""
    by_game = _records_by_game()
    if game_id not in by_game:
        return {"error": f"Game '{game_id}' not found. Call get_all_games to see valid IDs."}
    records = by_game[game_id]
    first = records[0]
    return {
        "game_id": game_id,
        "home_team": first["home_team"],
        "away_team": first["away_team"],
        "commence_time": first["commence_time"],
        "odds_by_book": {
            r["sportsbook"]: {
                "spread": r["markets"]["spread"],
                "moneyline": r["markets"]["moneyline"],
                "total": r["markets"]["total"],
                "last_updated": r["last_updated"],
            }
            for r in records
        },
    }


# ---------------------------------------------------------------------------
# Tool 3: calculate_game_metrics
# ---------------------------------------------------------------------------

def calculate_game_metrics(game_id: str) -> dict:
    """
    For a game, calculate: vig per book, implied probabilities, no-vig fair odds,
    and the best available line for each side/market.
    """
    by_game = _records_by_game()
    if game_id not in by_game:
        return {"error": f"Game '{game_id}' not found."}

    records = by_game[game_id]
    first = records[0]
    home = first["home_team"]
    away = first["away_team"]

    metrics_by_book = {}
    ml_home_lines: list[tuple[str, int]] = []
    ml_away_lines: list[tuple[str, int]] = []
    spread_home_lines: list[tuple[str, int]] = []
    spread_away_lines: list[tuple[str, int]] = []
    over_lines: list[tuple[str, int]] = []
    under_lines: list[tuple[str, int]] = []

    for r in records:
        book = r["sportsbook"]
        ml = r["markets"]["moneyline"]
        sp = r["markets"]["spread"]
        tot = r["markets"]["total"]

        ml_vig = calculate_vig(ml["home_odds"], ml["away_odds"])
        sp_vig = calculate_vig(sp["home_odds"], sp["away_odds"])
        tot_vig = calculate_vig(tot["over_odds"], tot["under_odds"])
        ml_fair = calculate_fair_odds(ml["home_odds"], ml["away_odds"])

        metrics_by_book[book] = {
            "moneyline": {
                "home_odds": ml["home_odds"],
                "away_odds": ml["away_odds"],
                "home_implied_prob": round(american_to_implied_prob(ml["home_odds"]), 4),
                "away_implied_prob": round(american_to_implied_prob(ml["away_odds"]), 4),
                "vig": round(ml_vig.vig, 4),
                "vig_pct": f"{ml_vig.vig * 100:.2f}%",
                "fair_home_odds": ml_fair.fair_odds_a,
                "fair_away_odds": ml_fair.fair_odds_b,
            },
            "spread": {
                "home_line": sp["home_line"],
                "home_odds": sp["home_odds"],
                "away_line": sp["away_line"],
                "away_odds": sp["away_odds"],
                "vig": round(sp_vig.vig, 4),
                "vig_pct": f"{sp_vig.vig * 100:.2f}%",
            },
            "total": {
                "line": tot["line"],
                "over_odds": tot["over_odds"],
                "under_odds": tot["under_odds"],
                "vig": round(tot_vig.vig, 4),
                "vig_pct": f"{tot_vig.vig * 100:.2f}%",
            },
        }

        ml_home_lines.append((book, ml["home_odds"]))
        ml_away_lines.append((book, ml["away_odds"]))
        spread_home_lines.append((book, sp["home_odds"]))
        spread_away_lines.append((book, sp["away_odds"]))
        over_lines.append((book, tot["over_odds"]))
        under_lines.append((book, tot["under_odds"]))

    best_ml_home = best_line_for_side(ml_home_lines)
    best_ml_away = best_line_for_side(ml_away_lines)
    best_sp_home = best_line_for_side(spread_home_lines)
    best_sp_away = best_line_for_side(spread_away_lines)
    best_over = best_line_for_side(over_lines)
    best_under = best_line_for_side(under_lines)

    return {
        "game_id": game_id,
        "home_team": home,
        "away_team": away,
        "metrics_by_book": metrics_by_book,
        "best_lines": {
            "moneyline": {
                home: {"book": best_ml_home[0], "odds": best_ml_home[1]},
                away: {"book": best_ml_away[0], "odds": best_ml_away[1]},
            },
            "spread": {
                f"{home} ({records[0]['markets']['spread']['home_line']})": {
                    "book": best_sp_home[0], "odds": best_sp_home[1]
                },
                f"{away} ({records[0]['markets']['spread']['away_line']})": {
                    "book": best_sp_away[0], "odds": best_sp_away[1]
                },
            },
            "total": {
                "over": {"book": best_over[0], "odds": best_over[1]},
                "under": {"book": best_under[0], "odds": best_under[1]},
            },
        },
    }


# ---------------------------------------------------------------------------
# Tool 4: detect_stale_lines
# ---------------------------------------------------------------------------

def detect_stale_lines(lag_minutes_threshold: float = 120.0) -> dict:
    """
    Detect sportsbooks with stale lines for any game.

    Staleness is measured relative to the most recently updated book for the
    same game. A line is stale if it's more than lag_minutes_threshold minutes
    behind the freshest line for that game.
    """
    by_game = _records_by_game()
    stale_flags: list[dict] = []

    for gid, records in by_game.items():
        timestamps = {
            r["sportsbook"]: datetime.fromisoformat(
                r["last_updated"].replace("Z", "+00:00")
            ).timestamp()
            for r in records
        }
        max_ts = max(timestamps.values())
        freshest_book = max(timestamps, key=timestamps.__getitem__)
        freshest_dt = datetime.fromtimestamp(max_ts, tz=timezone.utc).isoformat()

        for r in records:
            book = r["sportsbook"]
            ts = timestamps[book]
            lag_minutes = (max_ts - ts) / 60
            if lag_minutes > lag_minutes_threshold:
                stale_flags.append({
                    "game_id": gid,
                    "home_team": r["home_team"],
                    "away_team": r["away_team"],
                    "sportsbook": book,
                    "last_updated": r["last_updated"],
                    "freshest_book": freshest_book,
                    "freshest_update": freshest_dt,
                    "lag_minutes": round(lag_minutes, 1),
                    "lag_hours": round(lag_minutes / 60, 2),
                })

    return {
        "threshold_minutes": lag_minutes_threshold,
        "stale_count": len(stale_flags),
        "stale_lines": sorted(stale_flags, key=lambda x: x["lag_minutes"], reverse=True),
    }


# ---------------------------------------------------------------------------
# Tool 5: detect_outlier_prices
# ---------------------------------------------------------------------------

def detect_outlier_prices(std_dev_threshold: float = 2.0) -> dict:
    """
    Detect odds that are statistical outliers compared to consensus across books.

    Works in implied-probability space so comparisons are linear.
    Checks moneyline (home + away), spread (home + away), and total (over + under).
    """
    by_game = _records_by_game()
    outliers: list[dict] = []

    for gid, records in by_game.items():
        first = records[0]
        markets_to_check = [
            ("moneyline", "home_odds", first["home_team"]),
            ("moneyline", "away_odds", first["away_team"]),
            ("spread", "home_odds", f"{first['home_team']} spread"),
            ("spread", "away_odds", f"{first['away_team']} spread"),
            ("total", "over_odds", "over"),
            ("total", "under_odds", "under"),
        ]

        for market, field, label in markets_to_check:
            odds_by_book = {
                r["sportsbook"]: r["markets"][market][field]
                for r in records
            }
            analysis = detect_outliers(odds_by_book, std_dev_threshold)
            for book, info in analysis.items():
                if info["is_outlier"]:
                    outliers.append({
                        "game_id": gid,
                        "home_team": first["home_team"],
                        "away_team": first["away_team"],
                        "sportsbook": book,
                        "market": market,
                        "side": label,
                        "odds": info["odds"],
                        "implied_prob": info["implied_prob"],
                        "z_score": info["z_score"],
                        "consensus_odds": {
                            b: d["odds"] for b, d in analysis.items() if b != book
                        },
                    })

    return {
        "threshold_std_devs": std_dev_threshold,
        "outlier_count": len(outliers),
        "outliers": sorted(outliers, key=lambda x: abs(x["z_score"]), reverse=True),
    }


# ---------------------------------------------------------------------------
# Tool 6: detect_arbitrage_opportunities
# ---------------------------------------------------------------------------

def detect_arbitrage_opportunities(min_profit_pct: float = 0.0) -> dict:
    """
    Scan all games for cross-book arbitrage on moneyline markets.

    An arb exists when the sum of best implied probabilities across books < 1.0.
    Returns all opportunities with profit margin >= min_profit_pct (0–100 scale).
    """
    by_game = _records_by_game()
    opportunities: list[dict] = []

    for gid, records in by_game.items():
        first = records[0]
        home = first["home_team"]
        away = first["away_team"]

        ml_home = [(r["sportsbook"], r["markets"]["moneyline"]["home_odds"]) for r in records]
        ml_away = [(r["sportsbook"], r["markets"]["moneyline"]["away_odds"]) for r in records]

        result = detect_arbitrage(ml_home, ml_away)
        profit_pct = result.profit_margin * 100

        if result.exists and profit_pct >= min_profit_pct:
            # Calculate optimal stake split for $100 total stake
            stake_home = (american_to_implied_prob(result.best_side_a[1]) / 1.0) * 100
            stake_away = (american_to_implied_prob(result.best_side_b[1]) / 1.0) * 100
            total_stake = stake_home + stake_away
            norm_home = (stake_home / total_stake) * 100
            norm_away = (stake_away / total_stake) * 100

            opportunities.append({
                "game_id": gid,
                "home_team": home,
                "away_team": away,
                "market": "moneyline",
                "profit_pct": round(profit_pct, 2),
                "total_implied_prob": round(result.total_implied_prob, 4),
                "legs": [
                    {
                        "side": home,
                        "book": result.best_side_a[0],
                        "odds": result.best_side_a[1],
                        "implied_prob": round(american_to_implied_prob(result.best_side_a[1]), 4),
                        "stake_per_100": round(norm_home, 2),
                    },
                    {
                        "side": away,
                        "book": result.best_side_b[0],
                        "odds": result.best_side_b[1],
                        "implied_prob": round(american_to_implied_prob(result.best_side_b[1]), 4),
                        "stake_per_100": round(norm_away, 2),
                    },
                ],
            })

    opportunities.sort(key=lambda x: x["profit_pct"], reverse=True)
    return {
        "min_profit_threshold_pct": min_profit_pct,
        "opportunities_found": len(opportunities),
        "opportunities": opportunities,
    }


# ---------------------------------------------------------------------------
# Tool 7: rank_sportsbooks
# ---------------------------------------------------------------------------

def rank_sportsbooks() -> dict:
    """
    Rank all sportsbooks by a composite quality score.

    Score factors (max 100):
      - Vig efficiency (50 pts): lower average vig = higher score
      - Freshness (30 pts): fewer stale lines = higher score
      - Price accuracy (20 pts): fewer outlier prices = higher score
    """
    by_game = _records_by_game()

    # Collect per-book stats
    book_stats: dict[str, dict] = {}

    # Vig data
    for gid, records in by_game.items():
        for r in records:
            book = r["sportsbook"]
            if book not in book_stats:
                book_stats[book] = {
                    "vigs": [],
                    "stale_count": 0,
                    "outlier_count": 0,
                    "total_lines": 0,
                }
            ml = r["markets"]["moneyline"]
            sp = r["markets"]["spread"]
            tot = r["markets"]["total"]
            book_stats[book]["vigs"].append(calculate_vig(ml["home_odds"], ml["away_odds"]).vig)
            book_stats[book]["vigs"].append(calculate_vig(sp["home_odds"], sp["away_odds"]).vig)
            book_stats[book]["vigs"].append(calculate_vig(tot["over_odds"], tot["under_odds"]).vig)
            book_stats[book]["total_lines"] += 3

    # Staleness data
    stale_result = detect_stale_lines()
    for flag in stale_result["stale_lines"]:
        book = flag["sportsbook"]
        if book in book_stats:
            book_stats[book]["stale_count"] += 1

    # Outlier data
    outlier_result = detect_outlier_prices()
    for o in outlier_result["outliers"]:
        book = o["sportsbook"]
        if book in book_stats:
            book_stats[book]["outlier_count"] += 1

    # Build rankings
    rankings = []
    for book, stats in book_stats.items():
        avg_vig = statistics.mean(stats["vigs"]) if stats["vigs"] else 0
        composite = score_book(
            avg_vig=avg_vig,
            stale_count=stats["stale_count"],
            outlier_count=stats["outlier_count"],
            total_lines=stats["total_lines"],
        )
        rankings.append({
            "sportsbook": book,
            "composite_score": composite,
            "avg_vig": round(avg_vig, 4),
            "avg_vig_pct": f"{avg_vig * 100:.2f}%",
            "stale_lines": stats["stale_count"],
            "outlier_prices": stats["outlier_count"],
            "total_lines_evaluated": stats["total_lines"],
        })

    rankings.sort(key=lambda x: x["composite_score"], reverse=True)
    for i, r in enumerate(rankings):
        r["rank"] = i + 1

    return {"rankings": rankings}


# ---------------------------------------------------------------------------
# Tool 8: get_live_draftkings_odds
# ---------------------------------------------------------------------------

_DK_API_URL = (
    "https://sportsbook-nash.draftkings.com"
    "/api/sportscontent/dkusnj/v1/leagues/42648"
)


def get_live_draftkings_odds(nba_odds: dict | None = None) -> dict:
    """Fetch live NBA odds from the DraftKings public API.

    If *nba_odds* is provided it is used directly; otherwise the DK API is
    called.  On any failure the tool silently falls back to the cached
    DraftKings entries in odds.json.

    Returns structured moneyline, spread, and total odds per game.
    """
    import signal
    import urllib.request

    if nba_odds is not None:
        return _parse_draftkings_response(nba_odds)

    def _timeout_handler(signum, frame):
        raise TimeoutError("DK API request timed out")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    try:
        signal.alarm(5)  # hard 5-second deadline
        req = urllib.request.Request(
            _DK_API_URL,
            headers={
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (compatible; odds-sleuth/1.0)",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        signal.alarm(0)
        return _parse_draftkings_response(data)
    except Exception:
        signal.alarm(0)
        return _fallback_cached_dk_odds()
    finally:
        signal.signal(signal.SIGALRM, old_handler)


def _parse_american_odds_str(odds_str: str) -> int:
    """Convert DK display odds (unicode minus / plus) to an integer."""
    cleaned = odds_str.replace("\u2212", "-").replace("\u002B", "+").replace("−", "-")
    return int(cleaned)


def _parse_draftkings_response(data: dict) -> dict:
    """Transform a DK sportscontent JSON payload into a structured odds dict."""
    events = {e["id"]: e for e in data.get("events", [])}

    # Index markets by (eventId, marketTypeName)
    markets_by_event: dict[str, dict[str, dict]] = {}
    for m in data.get("markets", []):
        eid = m["eventId"]
        mtype = m["marketType"]["name"]             # Moneyline / Spread / Total
        markets_by_event.setdefault(eid, {})[mtype] = m

    # Index selections by marketId
    sels_by_market: dict[str, list[dict]] = {}
    for s in data.get("selections", []):
        sels_by_market.setdefault(s["marketId"], []).append(s)

    games: list[dict] = []
    for eid, event in events.items():
        participants = event.get("participants", [])
        home = next((p for p in participants if p.get("venueRole") == "Home"), None)
        away = next((p for p in participants if p.get("venueRole") == "Away"), None)
        if not home or not away:
            continue

        em = markets_by_event.get(eid, {})
        game: dict[str, Any] = {
            "event_id": eid,
            "name": event.get("name", ""),
            "start_time": event.get("startEventDate", ""),
            "home_team": home["name"],
            "away_team": away["name"],
            "markets": {},
        }

        # --- Moneyline ---
        if "Moneyline" in em:
            ml_sels = sels_by_market.get(em["Moneyline"]["id"], [])
            h = next((s for s in ml_sels if s.get("outcomeType") == "Home"), None)
            a = next((s for s in ml_sels if s.get("outcomeType") == "Away"), None)
            if h and a:
                game["markets"]["moneyline"] = {
                    "home_odds": _parse_american_odds_str(h["displayOdds"]["american"]),
                    "away_odds": _parse_american_odds_str(a["displayOdds"]["american"]),
                    "home_decimal": h.get("trueOdds"),
                    "away_decimal": a.get("trueOdds"),
                }

        # --- Spread ---
        if "Spread" in em:
            sp_sels = [s for s in sels_by_market.get(em["Spread"]["id"], []) if s.get("main")]
            h = next((s for s in sp_sels if s.get("outcomeType") == "Home"), None)
            a = next((s for s in sp_sels if s.get("outcomeType") == "Away"), None)
            if h and a:
                game["markets"]["spread"] = {
                    "home_line": h.get("points", 0),
                    "home_odds": _parse_american_odds_str(h["displayOdds"]["american"]),
                    "away_line": a.get("points", 0),
                    "away_odds": _parse_american_odds_str(a["displayOdds"]["american"]),
                }

        # --- Total ---
        if "Total" in em:
            tot_sels = [s for s in sels_by_market.get(em["Total"]["id"], []) if s.get("main")]
            ov = next((s for s in tot_sels if s.get("outcomeType") == "Over"), None)
            un = next((s for s in tot_sels if s.get("outcomeType") == "Under"), None)
            if ov and un:
                game["markets"]["total"] = {
                    "line": ov.get("points", 0),
                    "over_odds": _parse_american_odds_str(ov["displayOdds"]["american"]),
                    "under_odds": _parse_american_odds_str(un["displayOdds"]["american"]),
                }

        games.append(game)

    return {
        "source": "draftkings_live",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "games_count": len(games),
        "games": games,
    }


def _fallback_cached_dk_odds() -> dict:
    """Extract DraftKings rows from odds.json as a silent fallback."""
    data = _load_data()
    dk_records = [r for r in data["odds"] if r["sportsbook"] == "DraftKings"]

    games: list[dict] = []
    seen: set[str] = set()
    for r in dk_records:
        gid = r["game_id"]
        if gid in seen:
            continue
        seen.add(gid)
        games.append({
            "event_id": gid,
            "name": f"{r['away_team']} @ {r['home_team']}",
            "start_time": r["commence_time"],
            "home_team": r["home_team"],
            "away_team": r["away_team"],
            "markets": {
                "moneyline": {
                    "home_odds": r["markets"]["moneyline"]["home_odds"],
                    "away_odds": r["markets"]["moneyline"]["away_odds"],
                },
                "spread": {
                    "home_line": r["markets"]["spread"]["home_line"],
                    "home_odds": r["markets"]["spread"]["home_odds"],
                    "away_line": r["markets"]["spread"]["away_line"],
                    "away_odds": r["markets"]["spread"]["away_odds"],
                },
                "total": {
                    "line": r["markets"]["total"]["line"],
                    "over_odds": r["markets"]["total"]["over_odds"],
                    "under_odds": r["markets"]["total"]["under_odds"],
                },
            },
        })

    return {
        "source": "draftkings_cached",
        "note": "Live DraftKings API was unreachable; showing cached data from odds.json",
        "games_count": len(games),
        "games": games,
    }


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_all_games",
            "description": (
                "Returns a list of all NBA games in the dataset with their teams, "
                "game IDs, start times, and which sportsbooks have odds. "
                "Call this first to discover available game IDs before querying specific games."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_game_odds",
            "description": (
                "Returns the raw spread, moneyline, and total odds for a specific game "
                "across all sportsbooks, along with each book's last_updated timestamp."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "The game ID (e.g. 'nba_20260320_lal_bos'). Get valid IDs from get_all_games.",
                    }
                },
                "required": ["game_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_game_metrics",
            "description": (
                "Calculates derived metrics for a specific game: implied probabilities, "
                "vig/margin per sportsbook per market, no-vig fair odds, and the best "
                "available line for each side. Shows the math explicitly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "game_id": {
                        "type": "string",
                        "description": "The game ID to analyze.",
                    }
                },
                "required": ["game_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_stale_lines",
            "description": (
                "Scans all games and flags sportsbooks whose lines haven't been updated "
                "recently relative to other books for the same game. Staleness is measured "
                "as minutes behind the freshest book for that game."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "lag_minutes_threshold": {
                        "type": "number",
                        "description": (
                            "A book is considered stale if its last_updated is more than this "
                            "many minutes behind the freshest book for the same game. Default: 120."
                        ),
                        "default": 120,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_outlier_prices",
            "description": (
                "Detects odds that are statistical outliers compared to consensus across books. "
                "Uses z-score analysis in implied-probability space. Checks all markets "
                "(moneyline, spread, total) for each game."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "std_dev_threshold": {
                        "type": "number",
                        "description": (
                            "Number of standard deviations from the mean to flag as an outlier. "
                            "Default: 2.0 (catches ~5% of lines statistically)."
                        ),
                        "default": 2.0,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_arbitrage_opportunities",
            "description": (
                "Scans all games for cross-book moneyline arbitrage opportunities — situations "
                "where betting both sides across different books guarantees a profit regardless "
                "of outcome. Returns profit percentage and optimal stake split."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "min_profit_pct": {
                        "type": "number",
                        "description": (
                            "Minimum profit percentage to report (0–100 scale). "
                            "Default: 0 (return all arbs, even tiny ones)."
                        ),
                        "default": 0,
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_sportsbooks",
            "description": (
                "Ranks all sportsbooks by a composite quality score (max 100) based on: "
                "vig efficiency (50 pts), line freshness (30 pts), and price accuracy (20 pts). "
                "Useful for identifying which books to trust and which to avoid."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_draftkings_odds",
            "description": (
                "Fetches live NBA odds directly from the DraftKings sportsbook API. "
                "Returns current moneyline, spread, and total (over/under) odds for every "
                "game on tonight's slate. Call this as a FINAL step after rank_sportsbooks "
                "to publish the latest DraftKings lines in the briefing. "
                "Falls back silently to cached data if the API is unreachable."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ---------------------------------------------------------------------------
# Tool dispatcher — maps function name → implementation
# ---------------------------------------------------------------------------

TOOL_DISPATCH: dict[str, Any] = {
    "get_all_games": get_all_games,
    "get_game_odds": get_game_odds,
    "calculate_game_metrics": calculate_game_metrics,
    "detect_stale_lines": detect_stale_lines,
    "detect_outlier_prices": detect_outlier_prices,
    "detect_arbitrage_opportunities": detect_arbitrage_opportunities,
    "rank_sportsbooks": rank_sportsbooks,
    "get_live_draftkings_odds": get_live_draftkings_odds,
}


def dispatch_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name and return the result as a JSON string."""
    if name not in TOOL_DISPATCH:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = TOOL_DISPATCH[name](**arguments)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"Tool execution failed: {str(e)}"})
