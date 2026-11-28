- **P1 — Memory exhaustion** ([executor.py:117](/tmp/claude-1000/build4/executor.py:117)): `list(finditer(...))` retains every match before checking limits. Reproduced: `"1 " * 200000` (400 KB) allocates ~48 MB temporarily; larger inputs can OOM despite single-digit tokens. Retain only the last match while iterating; add a many-token resource test.

- **P2 — Direct execution skips 23 new tests** ([test_scoring.py:543](/tmp/claude-1000/build4/test_scoring.py:543)): `unittest.main()` exits before the appended test classes are defined, allowing a green run without their coverage. Move the main guard to EOF. Discovery is unaffected.

VERDICT: FIX FIRST