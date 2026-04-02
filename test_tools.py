"""Quick validation tests for the tools module."""
import json
from agent.tools import dispatch_tool, TOOL_DISPATCH, TOOL_SCHEMAS

passed = 0
failed = 0

def check(label, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS: {label}")
        passed += 1
    else:
        print(f"  FAIL: {label} — {detail}")
        failed += 1

# 1. Registration
print("=== Tool Registration ===")
schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
dispatch_names = set(TOOL_DISPATCH.keys())
check("schemas == dispatch", schema_names == dispatch_names,
      f"schemas={schema_names} dispatch={dispatch_names}")
check("8 tools registered", len(TOOL_SCHEMAS) == 8, f"got {len(TOOL_SCHEMAS)}")

# 2. Dispatch smoke tests
print("\n=== Dispatch Smoke Tests ===")
tests = [
    ("get_all_games", {}),
    ("get_game_odds", {"game_id": "nba_20260320_lal_bos"}),
    ("calculate_game_metrics", {"game_id": "nba_20260320_lal_bos"}),
    ("detect_stale_lines", {}),
    ("detect_outlier_prices", {}),
    ("detect_arbitrage_opportunities", {}),
    ("rank_sportsbooks", {}),
    ("get_live_draftkings_odds", {}),
]
for name, args in tests:
    result = dispatch_tool(name, args)
    parsed = json.loads(result)
    check(name, "error" not in parsed, parsed.get("error", ""))

# 3. DK tool output structure
print("\n=== DK Tool Output Validation ===")
dk = json.loads(dispatch_tool("get_live_draftkings_odds", {}))
check("has source", "source" in dk)
check("has games_count", "games_count" in dk)
check("has games list", "games" in dk)
check("games_count > 0", dk.get("games_count", 0) > 0, f"got {dk.get('games_count')}")
print(f"  (source={dk.get('source')}, count={dk.get('games_count')})")

for g in dk.get("games", []):
    name = g.get("name", "?")
    m = g.get("markets", {})
    check(f"{name} has moneyline", "moneyline" in m)
    check(f"{name} has spread", "spread" in m)
    check(f"{name} has total", "total" in m)
    if "moneyline" in m:
        ml = m["moneyline"]
        check(f"{name} ML home_odds is int", isinstance(ml.get("home_odds"), int),
              f"got {type(ml.get('home_odds'))}")
        check(f"{name} ML away_odds is int", isinstance(ml.get("away_odds"), int),
              f"got {type(ml.get('away_odds'))}")

# 4. Error handling
print("\n=== Error Handling ===")
err = json.loads(dispatch_tool("nonexistent_tool", {}))
check("unknown tool returns error", "error" in err)

# Summary
print(f"\n{'='*40}")
print(f"PASSED: {passed}  FAILED: {failed}")
if failed:
    exit(1)
print("ALL TESTS PASSED")
