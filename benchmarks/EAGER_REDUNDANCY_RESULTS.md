# Redundant-clause elimination in eager SAT (native CaDiCaL)

Date: 2026-10-13

## Correctness argument

The original eager encoding emits both `[-e[p,a,q],-e[p,a,r]]` and
`[-e[p,a,r],-e[p,a,q]]`. They are the same clause; retaining only
`q < r` is logically identical.

The explicit positive/negative incompatible input-state clauses
`[-x[p,i],-x[q,i]]` are **implied** by the two existing clauses
`[-x[p,i],final[i]]` and `[-x[q,i],-final[i]]` via resolution. Omitting
them preserves satisfiability, but may weaken propagation.

The four experiments are `ordered` (only eliminate symmetric duplicate
at-most-one clauses), `implied` (only omit resolvable residual-pair clauses),
`both`, and `dedup` (`both` plus general canonical deduplication).
All variants keep the original eager constraints otherwise unchanged.
No change to the default solver is proposed here.

## Native benchmarks (single measurements)

Run: https://github.com/liyong31/DFAMiner/actions/runs/38402749763

Python 3.11, PySAT CaDiCaL 1.9.5, same GitHub Actions runner.
Each case checks one target size below the known or generating optimum.
Times include Python CNF generation and native solving. All completed
results returned UNSAT. A reported TIMEOUT means 18 s expired.

| 3DFA states / k | Baseline | Ordered | Implied | Both | Both + dedup |
|---|---:|---:|---:|---:|---:|
| 348 / 6 | 2.790s | 1.669s | **1.616s** | 1.820s | 1.840s |
| 392 / 7 | 14.986s | **6.440s** | TIMEOUT | 15.066s | 15.103s |
| 465 / 8 | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT |
| 1554 / 5 | 2.107s | 1.784s | **1.607s** | 1.899s | 1.970s |
| 3884 / 5 | 3.948s | 3.772s | 4.383s | **3.563s** | 3.913s |

## Clause counts and solver search

| Case | Baseline clauses | Ordered clauses | Both clauses | Baseline conflicts | Ordered conflicts |
|---|---:|---:|---:|---:|---:|
| 348 / 6 | 26,016 | 25,836 | 21,624 | 73,682 | 48,554 |
| 392 / 7 | 41,412 | 41,118 | 33,264 | 280,777 | 147,116 |
| 1554 / 5 | 53,955 | 53,855 | 53,655 | 47,009 | 39,459 |
| 3884 / 5 | 181,920 | 181,820 | 152,120 | 37,713 | 33,601 |

**Important:** The large 392-state speedup is not caused by substantial
clause-count reduction: only 294 clauses are omitted, yet CaDiCaL's
search changes and conflict count roughly halves. This needs repeated
runs with controlled random seeds / independent instances before generalising.

Conversely, on the same instance removing ~8,000 further implied
incompatibility clauses causes a timeout. Thus `ordered` is the
best conservative option when prioritising speed.

General deduplication increased construction overhead without changing
the SAT clause count on these examples.

## Recommendation

Prefer the `ordered` variant for continued evaluation and retain the
positive/negative incompatibility clauses as useful propagation lemmas.
Do not automatically remove *all* logically implied clauses.

Before merging into the stable `cegar` branch, run correctness tests
for SAT/UNSAT equivalence on random automata, BFS symmetry-breaking
variants, and safety-automaton constraints. Larger parity-game instances
and solver seeds should be evaluated independently.
