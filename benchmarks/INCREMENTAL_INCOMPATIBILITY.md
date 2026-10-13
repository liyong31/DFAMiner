# Incremental pairwise incompatibility during 3DFA construction

The experimental collector is attached to the existing lexical, bottom-up
3DFA builder only when the opt-in flag is enabled.

For canonical registered states p and q:

p incompatible q iff (opposite positive/negative labels) or
there is a common defined letter a whose canonical successor pair is
already incompatible.

Children are registered before parents. The exact relation is stored in
triangular Python-integer bitsets and exported to final 3DFA indices after
conversion. A deterministic multi-seed greedy clique restricted to
root-reachable states provides a certified lower bound on target DFA size.

Eager SAT may also add a bounded selection of clauses
(-x[p,i] or -x[q,i]) for selected incompatible p,q. These are implied
by the original eager consistency constraints and preserve exactness.
The existing ordered eager encoding remains the default in the opt-in mode.

## Native validation

Workflow: https://github.com/liyong31/DFAMiner/actions/runs/38406133052

Three test families passed, checking random input words, independent
product-BFS pairwise incompatibility, graph symmetry and clique validity,
and SAT equivalence with the baseline on small automata.

## Collection overhead and lower bounds

| Source | 3DFA states | Canonical pair checks | Baseline build | With collection | Clique lower bound |
|---|---:|---:|---:|---:|---:|
| repo4 | 210 | 21,945 | 0.0021 s | 0.0134 s | 4 |
| repo7 | 348 | 60,378 | 0.0034 s | 0.0327 s | 7 |
| repo8 | 392 | 76,636 | 0.0038 s | 0.0425 s | 6 |
| repo10 | 465 | 107,880 | 0.0049 s | 0.0603 s | 6 |
| repo12 | 551 | 151,525 | 0.0058 s | 0.0838 s | 8 |
| sparse600 | 1,554 | 1,206,681 | 0.0122 s | 0.4415 s | 3 |
| sparse2400 | 3,884 | 7,540,786 | 0.0416 s | 3.4068 s | 5 |

The clique of size 7 on repo7 matches its previously reported optimum,
eliminating all UNSAT target sizes 1 through 6 when using the full
minimisation command. This is a lower bound from a checkable certificate,
not an improvement for an isolated k=6 SAT call.

On repo8, repo10 and large sparse cases, clique bounds are weaker than
known optima. Full pair collection costs 3.4s on the 3,884-state
synthetic case: its preprocessing overhead could outweigh solving gains.

## Follow-up experiments

Native SAT ablation: benchmarks/compare_incompatibility_sat.py and
.github/workflows/incompatibility-sat.yml. This compares ordered eager
with no pair clauses, clique pairs, and 100 or 1000 additional pairs.
Do not enable all-pair clauses indiscriminately because they may be
quadratic in input states times target DFA states.

Research next step: compute a faster strong clique / bound without full
O(N^2) pair materialisation, e.g. backward conflict propagation from
labels or selective residual-pair discovery.
