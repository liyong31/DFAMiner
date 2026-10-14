# Cheap certified lower bound for eager DFA inference

Run: https://github.com/liyong31/DFAMiner/actions/runs/38412264285
Date: 13 October 2026. Native PySAT/CaDiCaL 1.9.5.

The opt-in --cheap-bound mode uses a time-limited sample of at most
32 distinguishing suffixes (defaults), up to 256 reachable residual-state
candidates, and greedy cliques of *witnessed* incompatibilities. The
--cheap-bound-ms flag controls the soft collection limit (default 25 ms).

A clique certificate consists of input states and a concrete suffix for
each clique edge; independent validation checks every witness.
The algorithm does not construct the quadratic residual-pair matrix.

## Native lower bounds

| Example | 3DFA states | Certified lower bound | Typical 10-25ms-config runtime |
|---|---:|---:|---:|
| repo7 | 348 | 7 | 12-28 ms |
| repo8 | 392 | 6 | 6-7 ms |
| repo10 | 465 | 6 | 6-7 ms |
| sparse2400 | 3,884 | 4 | 9 ms |
| sparse4800 | 6,282 | 4 | 9 ms |

## End-to-end ordered eager solving

Each case was built once and minimised using ordered eager SAT, with no
extra incompatibility clauses. Times include CNF construction and SAT
solving (but exclude the shared input construction). Correct minimum
DFA sizes were independently checked by product verification.

| Case | Without cheap bound | With cheap bound | Benefit |
|---|---:|---:|---|
| repo7 | 2.0046 s | 0.7290 s | ~2.75x faster |
| sparse2400 | 4.9083 s | 5.0464 s | ~2.8% slower |
| sparse4800 | 4.4469 s | 4.2023 s | ~5.5% faster |

The repo7 bound 7 is exact and skips k=1,...,6, giving a real win.
The large synthetic cases get bound 4 rather than the true optimum 6.
Skipping k=1,2,3 brings little benefit because those SAT UNSAT proofs
are inexpensive. Nonetheless, the bound is cheap (~9ms) and avoids the
multi-second overhead of full pairwise collection.

These are single measurements. On large examples the net time
difference is small and may fluctuate with machine noise and SAT
heuristics. Do not treat this as proof of a large general speedup.

Next: Improve suffix selection while respecting strict millisecond budgets,
and compare with richer but inexpensive local distinguishing-suffix
families. Keep ordered eager SAT as the central solver.
