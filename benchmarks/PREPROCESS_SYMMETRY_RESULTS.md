# Low-cost 3DFA preprocessing and symmetry experiments

Native benchmark:
https://github.com/liyong31/DFAMiner/actions/runs/38563530651
Native differential correctness:
https://github.com/liyong31/DFAMiner/actions/runs/38563948123

All finished comparisons used PySAT CaDiCaL 1.9.5 and the compact
eager encoder (remove redundant explicit opposite-labelled state-pair
clauses). Baseline is identical except for the experimental feature.

Source trim: forward reachable AND backwards-to-labelled, retaining
all input initial states. No test automaton had any removable states.
The trim option did not help these builder-canonicalised 3DFAs.
It may help arbitrary externally supplied 3DFAs with unobserved states.

Input-order symmetry: restrict target-state numbering to first
appearances among the first 32 rooted source states. Requires no
subquadratic or quadratic conflict graph. Root-reachable input states
have exactly one target assignment, by deterministic consistency.
The resulting restricted-growth clauses are invariant under target
state renaming, so do not affect existence of a consistent target.
They are NOT combined with existing target-BFS symmetry breaking.

| 3DFA states | k UNSAT | Compact eager | Input RGS BFS | Target BFS |
|---:|---:|---:|---:|---:|
| 348 | 6 | 1.713 s | 0.084 s | 0.063 s |
| 392 | 7 | 13.951 s | 0.510 s | 0.236 s |
| 465 | 8 | >25 s | 1.770 s | 0.610 s |
| 3,884 | 5 | 3.479 s | 1.017 s | 0.514 s |
| 6,282 | 5 | 2.379 s | 1.622 s | not tested |
| 10,107 | 5 | 7.928 s | 1.919 s | not tested |

All completed variants returned UNSAT. The existing target-BFS
constraints won on every medium case tested, despite imposing more
structural constraints on target transitions.

On the 10,107-state case RGS adds only 93 clauses and reduces
CaDiCaL conflicts from 15,931 to 2,684. Target-BFS was not compared
at that size.

The current benchmark deliberately used no target-BFS for compact
baseline, as earlier benchmarks did. Given the results, benchmark
protocols and defaults should explicitly compare BFS=true and false
rather than assuming disabling it is favourable.

Random cyclic automata and a multiple-root automaton were tested for
SAT equivalence of input-RGS and trim against compact eager at k=1,2,3.
No discrepancies; tests passed.

Next high-value priorities:
1. Benchmark existing target-BFS on 6k, 10k, 16k source states and SAT
   k=6, not just UNSAT k=5.
2. Check whether the target-BFS implementation can be simplified or
   compiled more compactly.
3. Enable/disable symmetry based on independent benchmark evidence,
   never by combining conflicting canonical numbering constraints.
4. Analyse source-state ordering and choose lightweight RGS prefixes
   only if target-BFS overhead becomes significant.
5. Add trimming as optional preprocessing for arbitrary input 3DFAs,
   not as a default for already canonicalised acyclic sample 3DFAs.

Single-run timings are sensitive to clause order and SAT heuristics;
replicate before committing to a default change.
