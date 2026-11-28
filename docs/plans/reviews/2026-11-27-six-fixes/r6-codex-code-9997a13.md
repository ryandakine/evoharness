- **P2 — `executor.py:numeric_key`: accepted answers split into separate vote groups.** `numeric_match("0.333333", "1/3") == 1`, but their keys differ. Verified: votes `[7, 1/3, 7, 0.333333, 0.3333333]` elect incorrect `7` despite three accepted answers. The nine-decimal test misses this mismatch. Define consistent scoring/grouping semantics and add this regression.

VERDICT: FIX FIRST