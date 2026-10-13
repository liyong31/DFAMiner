# Rollbackable CDCL product propagation — native results

Run: https://github.com/liyong31/DFAMiner/actions/runs/38377131386
Environment: Python 3.11, native PySAT/CaDiCaL 1.9.5 on GitHub Actions.
All original CDCL tests and three incremental theory tests passed, including
differential reachability checks following nested CDCL backtracking.

## Summary

Implemented trail-indexed reachable product pairs, per-letter reverse
listeners, rollback of all newly reached pairs, positive/negative label
witness lists per target state, and reasons reconstructed from predecessor
edges. A separate `--cdcl-incremental` flag selects the experimental backend
with `--cdcl-product --solver cadical195`. Existing behaviour is unchanged.

| Instance; k | Eager | CEGAR | Old CDCL | Rollback CDCL |
|---|---:|---:|---:|---:|
| Bundled 3DFA; k=2 UNSAT | — | 0.25 ms | 0.38 ms | 0.46 ms |
| Bundled 3DFA; k=3 SAT | — | 0.46 ms | 1.07 ms | 1.64 ms |
| Priority3 depth6; k=4 UNSAT | 7.58 ms | 2.28 ms | 9.3 ms | 35.5 ms |
| Priority3 depth7; k=4 UNSAT | 6.25 ms | 2.65 ms | 9.8 ms | 44.6 ms |
| Modulo4 depth8; k=3 UNSAT | 0.66 ms | 0.51 ms | 0.6 ms | 1.6 ms |

Selected native-run observations (one representative sample per case,
different cases reflect GitHub Actions timing noise). The priority examples
are synthetic finite-word priority patterns, not actual parity-game graphs.
All solvers agreed on SAT/UNSAT outcomes.

On priority3 depth7, the incremental theory traversed 16,989 discovered
product pairs across assignments, performed 1,359 partial checks, and found
644 early conflicts; original CDCL performed 371 checks and found 172 early
conflicts. Product reachability maintenance reduces redundant graph scans
but alters search order, increasing CDCL conflicts and suffix work.

**Conclusion: do not merge for performance.** Next experiment: keep
incremental reachability but retain the old propagator's conflict ordering
and lookahead selection. Alternatively move hot path into C++ and compare
with established CEGAR. Avoid claims of speedup on these workloads.
