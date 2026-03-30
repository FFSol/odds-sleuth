"""
Odds math utilities. All calculations are deterministic — no LLM involved.
These are called directly by tool implementations so the agent's math is always correct.
"""

from __future__ import annotations
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Core conversions
# ---------------------------------------------------------------------------

def american_to_implied_prob(odds: int) -> float:
    """Convert American odds to implied probability (0–1 range).

    Negative odds: |odds| / (|odds| + 100)
    Positive odds: 100 / (odds + 100)

    Examples:
        -150 → 0.6 (60%)
        +200 → 0.333 (33.3%)
        -110 → 0.5238 (52.38%)
    """
    if odds < 0:
        abs_odds = abs(odds)
        return abs_odds / (abs_odds + 100)
    else:
        return 100 / (odds + 100)


def implied_prob_to_american(prob: float) -> int:
    """Convert implied probability back to American odds (rounded to nearest integer)."""
    if prob >= 0.5:
        return round(-(prob / (1 - prob)) * 100)
    else:
        return round(((1 - prob) / prob) * 100)


# ---------------------------------------------------------------------------
# Vig / margin
# ---------------------------------------------------------------------------

class VigResult(NamedTuple):
    vig: float          # e.g. 0.0476 = 4.76%
    prob_side_a: float  # raw implied prob for side A
    prob_side_b: float  # raw implied prob for side B
    total_prob: float   # sum of both sides (> 1.0 due to vig)


def calculate_vig(odds_a: int, odds_b: int) -> VigResult:
    """Calculate the vig/margin for a two-sided market.

    Vig = sum of implied probs − 1.
    Example: -110/-110 → 52.38% + 52.38% = 104.76% → vig = 4.76%
    """
    prob_a = american_to_implied_prob(odds_a)
    prob_b = american_to_implied_prob(odds_b)
    total = prob_a + prob_b
    vig = total - 1.0
    return VigResult(vig=vig, prob_side_a=prob_a, prob_side_b=prob_b, total_prob=total)


# ---------------------------------------------------------------------------
# No-vig fair odds
# ---------------------------------------------------------------------------

class FairOdds(NamedTuple):
    fair_prob_a: float  # normalized to sum to 1
    fair_prob_b: float
    fair_odds_a: int    # American
    fair_odds_b: int    # American


def calculate_fair_odds(odds_a: int, odds_b: int) -> FairOdds:
    """Normalize implied probabilities to remove the vig, giving fair odds."""
    vig_result = calculate_vig(odds_a, odds_b)
    total = vig_result.total_prob
    fair_a = vig_result.prob_side_a / total
    fair_b = vig_result.prob_side_b / total
    return FairOdds(
        fair_prob_a=fair_a,
        fair_prob_b=fair_b,
        fair_odds_a=implied_prob_to_american(fair_a),
        fair_odds_b=implied_prob_to_american(fair_b),
    )


# ---------------------------------------------------------------------------
# Best available line
# ---------------------------------------------------------------------------

def best_line_for_side(odds_list: list[tuple[str, int]]) -> tuple[str, int]:
    """Given [(book, odds), ...], return the book+odds with highest payout.

    Highest payout = lowest implied probability = best odds for the bettor.
    For negative odds: least negative (e.g. -105 beats -115).
    For positive odds: most positive (e.g. +115 beats +105).
    In mixed: any positive beats any negative.
    """
    return max(odds_list, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Arbitrage detection
# ---------------------------------------------------------------------------

class ArbitrageResult(NamedTuple):
    exists: bool
    total_implied_prob: float   # < 1.0 means arb opportunity exists
    profit_margin: float        # e.g. 0.02 = 2% guaranteed profit
    best_side_a: tuple[str, int] | None  # (book, odds)
    best_side_b: tuple[str, int] | None


def detect_arbitrage(
    odds_side_a: list[tuple[str, int]],
    odds_side_b: list[tuple[str, int]],
) -> ArbitrageResult:
    """Detect if a cross-book arbitrage exists for a two-sided market.

    An arb exists when:
        best_implied_prob(A) + best_implied_prob(B) < 1.0

    The profit margin = 1 − total_implied_prob.
    """
    best_a = best_line_for_side(odds_side_a)
    best_b = best_line_for_side(odds_side_b)
    prob_a = american_to_implied_prob(best_a[1])
    prob_b = american_to_implied_prob(best_b[1])
    total = prob_a + prob_b
    exists = total < 1.0
    profit = (1.0 - total) if exists else 0.0
    return ArbitrageResult(
        exists=exists,
        total_implied_prob=total,
        profit_margin=profit,
        best_side_a=best_a,
        best_side_b=best_b,
    )


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

from datetime import datetime, timezone


def minutes_since_update(last_updated: str) -> float:
    """Return minutes elapsed since last_updated (ISO 8601 string)."""
    dt = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    return (now - dt).total_seconds() / 60


def is_stale(last_updated: str, threshold_minutes: float = 60.0) -> bool:
    """Return True if the line hasn't been updated within threshold_minutes."""
    return minutes_since_update(last_updated) > threshold_minutes


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

import statistics


def detect_outliers(
    odds_by_book: dict[str, int],
    std_dev_threshold: float = 2.0,
) -> dict[str, dict]:
    """Flag odds that deviate more than N standard deviations from the mean.

    Returns a dict of {book: {odds, z_score, is_outlier}} for every book.
    """
    values = list(odds_by_book.values())
    if len(values) < 3:
        return {book: {"odds": o, "z_score": 0.0, "is_outlier": False}
                for book, o in odds_by_book.items()}

    # Work in implied-prob space so the comparison is linear
    probs = [american_to_implied_prob(o) for o in values]
    mean = statistics.mean(probs)
    stdev = statistics.stdev(probs)

    result = {}
    for book, odds in odds_by_book.items():
        prob = american_to_implied_prob(odds)
        z = (prob - mean) / stdev if stdev > 0 else 0.0
        result[book] = {
            "odds": odds,
            "implied_prob": round(prob, 4),
            "z_score": round(z, 3),
            "is_outlier": abs(z) >= std_dev_threshold,
        }
    return result


# ---------------------------------------------------------------------------
# Sportsbook quality scoring
# ---------------------------------------------------------------------------

def score_book(
    avg_vig: float,
    stale_count: int,
    outlier_count: int,
    total_lines: int,
) -> float:
    """Composite quality score for a sportsbook (higher = better, max ~100).

    Weights:
      - Vig efficiency:    50 pts (lower vig = higher score)
      - Freshness:         30 pts (fewer stale lines = higher score)
      - Price accuracy:    20 pts (fewer outliers = higher score)
    """
    # Vig component: 0% vig → 50 pts, 10% vig → 0 pts
    vig_score = max(0.0, 50.0 * (1 - avg_vig / 0.10))

    # Freshness component
    stale_rate = stale_count / max(total_lines, 1)
    freshness_score = 30.0 * (1 - stale_rate)

    # Accuracy component
    outlier_rate = outlier_count / max(total_lines, 1)
    accuracy_score = 20.0 * (1 - outlier_rate)

    return round(vig_score + freshness_score + accuracy_score, 1)
