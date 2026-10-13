# Eager minimiser: next optimisation candidates (13 October 2026)

The current eager solver is in `minimiser.py`. Its CNF consists of target
transition variables e, target finality variables, and product reachability
variables x. The full encoding is polynomial but expensive near the
minimum when proving UNSAT. All proposals below require independently
verified SAT/UNSAT comparisons; avoid inferring speedup from clause counts.

## Priority 1: distinguish clause ordering from eliminating duplicates

The `ordered` variant in `eager_cnf_variants.py` changes both CNF clause
order and clause multiset. SAT heuristics may be sensitive to order.

Run a factorial ablation with:
- original clauses and original order
- remove only symmetric duplicates, preserving first occurrence/order
- preserve original multiset, reorder to match the reduced generator
- reduced generator as already benchmarked
- duplicate-free pairs plus sorted/structural clause phases

Collect CaDiCaL conflicts, decisions, propagations, total time, and
build time. Use 5+ repeats and separate samples for generalisation.
The 392-state instance is the first target; 465 states at k=8 is the
harder follow-up.

## Priority 2: structural preprocessing with language preservation

Compute root reachability and co-reachability to a labelled state.
Input states unable to influence any accepted/rejected word can be
discarded, along with transition constraints that lead only to them.
This is safe for general 3DFAs if done carefully with multiple roots,
cycles, and empty-word labels. It must not eliminate the initial state
when it has no informative reachable word: that is a trivial instance.

The existing `sdfa.reduce()` implements forward/backward trimming,
but is not invoked by eager `solve()`. Check its handling of rootless
outputs before using it as a preprocessing step. For large samples,
compare resulting N and m before/after.

## Priority 3: state incompatibility reasoning

Compute `p != q` when a common suffix leads one residual to a positive
state and the other to a negative state. A product BFS from all
oppositely-labelled pairs through reverse letter transitions computes
this in worst case O(N^2 |Sigma|), with sparse alternatives on acyclic
automata. Each pair implies for every target state i:
    not x[p,i] or not x[q,i].
These constraints are valid even with suffix sharing. They can improve
propagation, but adding *all* of them may be huge. Start with a maximal
clique/greedy colouring lower-bound certificate and selectively add
witnessed pairs that participate in difficult paths.

A clique with h pairwise incompatible *root-reachable* residuals
certifies minimum target size >= h. Never use incompatible states
unreachable from all initial states for this bound.

## Priority 4: symmetry-breaking variants

The baseline has BFS-tree constraints, switched off with `nobfs`.
Its auxiliary variables are allocated even when disabled, but with no
clauses they cause no substantial SAT search; compacting them is mainly
a memory/build concern. Compare:
- no BFS, with BFS
- canonical BFS with only essential tree clauses
- prefix-access representative symmetry breaking (must prove no valid
  target automaton is excluded)

Measure additional CNF and UNSAT conflicts; symmetry breaking can help
or harm depending on input.

## Priority 5: incremental target sizes and solver portfolio

Instead of rebuilding unrelated SAT instances for each k, use a single
upper-bound-k transition encoding with assumption-controlled activation
of states. Correctness and symmetry breaking need fresh proof, because
disabling target states cannot invalidate the 3DFA reachability witness
condition. This is a substantial redesign; start with independent k
experiments and alternative solver seeds (same strict timeout) first.

## Priority 6: propagation-oriented CNF

Explore strengthening with carefully selected redundant implications:
output incompatibility is already useful, as removing it caused an
18-second timeout on repo8 k=7. Try short distinguishing-suffix clauses
or pairwise propagation selected by heuristics, rather than indiscriminate
all-pair closure. At-most-one transition pair duplicates should be
filtered once, not repeatedly by generic full-CNF hashing.

## Recommended execution sequence

1. Clause-order ablation: isolate the cause of the observed 2.33x speedup.
2. Reachability/coreachability preprocessing with property-based tests.
3. Partial incompatibility graph + certified clique lower bound.
4. BFS vs no-BFS and phase/seed portfolio on repo8/repo10.
5. Consider variable ordering or alternative SAT encodings only with
   robust native CaDiCaL results.

All experiments should use the original 392-state k=7 and 465-state
k=8 UNSAT inputs, and multiple unrelated DFAMiner datasets to check
that a tuning does not overfit one sample.
