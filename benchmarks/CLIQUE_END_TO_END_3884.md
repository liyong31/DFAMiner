# One-minute end-to-end benchmark: 3,884-state 3DFA

**Date:** 2026-10-13. **Native run:**
https://github.com/liyong31/DFAMiner/actions/runs/38407627320

Four separate processes each received a 60-second wall-clock limit.
Every process finished and independently obtained minimum DFA size **6**.
The input is the deterministic sparse2400 sample generator (2,400
labelled words; 3,884 states; 6,049 input transitions).
Measured times cover sample/3DFA construction, incompatibility
collection (where applicable), per-k CNF construction and CaDiCaL 1.9.5
search. All satisfying candidates passed complete product verification.

| Configuration | k attempted | Build (s) | CNF (s) | SAT (s) | Total (s) |
|---|---|---:|---:|---:|---:|
| Original eager | 1,2,3,4,5,6 | 0.063 | 0.387 | 6.790 | 7.275 |
| Ordered eager | 1,2,3,4,5,6 | 0.048 | 0.359 | 5.812 | 6.260 |
| Clique lower bound only | 5,6 | 3.310 | 0.221 | 5.546 | 9.106 |
| Clique bound + 1,000 extra pairs | 5,6 | 3.354 | 0.240 | 1.468 | 5.094 |

The verified clique has five vertices, so a DFA of size <5 is
impossible. However, ordered eager SAT spent only about 0.357s of
search time at k=1,2,3,4 (about 0.485s including the respective
CNF construction). A full 3.3s quadratic pair collector cannot pay
for itself by skipping those sizes alone.

The **pair clauses** are the significant contribution to runtime:
- k=5 UNSAT: ordered eager 3.607s vs 1.051s with pair clauses.
- k=6 SAT: ordered eager 1.848s vs 0.417s with pair clauses.

The 1,000-pair budget is *additional* to the ten clique edges:
1,010 distinct selected incompatible pairs were injected. Thus, the
k=5 formula has 5,050 additional clauses; k=6 has 6,060.

**Conclusion:** clique lower bound alone is a net regression for the
3,884-state case; collecting pairs **and** using selected incompatibility
clauses gives 1.43x end-to-end improvement over original eager and
1.23x over ordered eager. Single-run timings can vary with SAT
heuristics; repeat across several independent examples before adopting
this configuration by default.
