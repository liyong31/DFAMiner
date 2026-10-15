# Native CDCL product performance study — 2026-10-13

Using Python 3.11 and native PySAT CaDiCaL 1.9.5 on GitHub Actions, all five
CDCL theory tests passed. The benchmark compared eager, CEGAR, and CDCL
on modular-language, finite priority-pattern, and random-DFA examples.
All solvers agreed on satisfiability for the same target size k.

## Representative timings (milliseconds)

| Case and k | Eager | CEGAR | Original CDCL | Modified CDCL |
|---|---:|---:|---:|---:|
| priority3x6, k=3 UNSAT | 2.24 | 1.47 | 3.01 | 3.09 |
| priority3x6, k=4 UNSAT | 10.48 | 3.30 | 18.49 | 18.89 |
| priority3x7, k=3 UNSAT | 3.06 | 1.82 | 3.96 | 3.22 |
| priority3x7, k=4 UNSAT | 9.72 | 4.11 | 19.69 | 20.53 |
| random5x8x17, k=4 SAT | 1.65 | 1.27 | 1.14 | 1.14 |

Times are single repeats, for orientation only. Small cases are sensitive
to machine noise. The generated finite priority words are not real game graphs.

## Theory scan sweep — priority3x7 at k=4, UNSAT

| Scan frequency | Lookahead probes | CDCL ms | Partial scans |
|---:|---:|---:|---:|
| 1 | 0 | 41.42 | 1080 |
| 1 | 32 | 17.88 | 368 |
| 4 | 0 | 35.49 | 867 |
| 4 | 32 | 20.57 | 402 |
| 16 | 0 | 41.21 | 707 |
| 16 | 32 | 22.67 | 395 |
| 64 | 0 | 47.63 | 707 |
| 64 | 32 | 22.37 | 395 |

With lookahead, frequency=1 improves over frequency=4, but is still
substantially slower than CEGAR (~4.1 ms). Disabling distinguishing suffix
lookahead increases both theory scans and total time.

## Changes tested

Incrementally cache selected target transitions, rebuilding only after
backtracking. Attempt cheap accepting-label implications before suffix
lookahead. Five native theory tests and same-size SAT/UNSAT comparisons pass.

Performance is mixed and does not justify merging this patch into the
stable cegar branch yet.

## Next direction

Replace Python repeated full partial-product traversal with reversible,
trail-indexed incremental reachability, and cache distinguishing-suffix
certificates. Benchmark against CEGAR on real larger parity-game 3DFAs.
