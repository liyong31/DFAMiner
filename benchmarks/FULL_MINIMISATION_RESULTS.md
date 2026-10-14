# Complete minimum-DFA benchmark: k=1 versus sound clique bound

Native GitHub Actions: https://github.com/liyong31/DFAMiner/actions/runs/38567668528

2026-10-14. All configurations use compact eager CNF, target-BFS symmetry
breaking, and CaDiCaL 1.9.5. The comparison covers *complete* minimisation
from k=1 or from a separately verified clique bound, stopping at the
first satisfiable DFA. Each new k uses a fresh SAT solver. All SAT
candidates were checked independently against the original 3DFA.
The known minimum is checked after solving, not used to choose the start.

Results include input construction, preprocessing, CNF, solver bootstrap,
SAT and verification. Single measurements only; differences below 10%
may be noise. Values in seconds.

| Case | OPT | Start 1, compact BFS | Certified bound only | Degree300 from 1 | Degree300 from bound | Short300 from bound |
|---|---:|---:|---:|---:|---:|---:|
| repo7 / 348 | 7 | 0.26155 | 0.16971 | 0.18067 | **0.09462** | 0.09977 |
| repo8 / 392 | 8 | 0.81882 | 0.72687 | 0.30207 | **0.25326** | 0.42356 |
| repo10 / 465 | 9 | 2.38692 | 2.35232 | 0.73881 | **0.68373** | 2.14073 |
| sparse2400 / 3884 | 6 | 1.23865 | **1.17285** | not run | 1.20915 | 1.30826 |
| sparse4800 / 6282 | 6 | 2.41046 | 2.21301 | not run | 2.40023 | **1.95397** |
| sparse9600 / 10107 | 6 | 4.58209 | **4.21290** | not run | 4.21733 | 9.49755 |

| Case | Witnessed clique | Default k calls | Bound k calls |
|---|---:|---:|---:|
| repo7 | 7 | 1..7 (7) | 7 (1) |
| repo8 | 6 | 1..8 (8) | 6..8 (3) |
| repo10 | 6 | 1..9 (9) | 6..9 (4) |
| sparse2400 | 4 | 1..6 (6) | 4..6 (3) |
| sparse4800 | 4 | 1..6 (6) | 4..6 (3) |
| sparse9600 | 4 | 1..6 (6) | 4..6 (3) |

The lower bound by itself saves little SAT search on repo10 because
smaller k are cheap compared with k=8/9; degree-scored clauses explain
most of the observed speedup.

On the large sparse cases, bound-only tends to win slightly but
preprocessing is cheap and the evidence is not replicated. Short-scored
clauses can substantially regress on the 10,107-state example.

Recommendation: compact eager + target BFS as baseline; cheap sound
lower bound as an optional starting-point accelerator; degree-scored
lemmas selectively on hard examples, not as an unconditional default.
Replicate across seeds and run other families before deciding a
universal adaptive policy.

Implementation:
`benchmarks/compare_full_minimisation.py`
