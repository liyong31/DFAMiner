# Larger 3DFA end-to-end scaling: 4,800 samples, 6,282 states

Native run: https://github.com/liyong31/DFAMiner/actions/runs/38408916791

60-second time limit **per method**. All methods independently found and
verified the minimum six-state DFA, using the same deterministic sample
generator. The sample count was explicitly corrected to pass through
subprocesses; an earlier workflow run mistakenly repeated the 2,400-sample
case and is **not** part of this comparison.

| Method | Starting k | Collection / build (s) | CNF (s) | SAT (s) | Total (s) |
|---|---:|---:|---:|---:|---:|
| Original eager | 1 | 0.108 | 1.183 | 6.501 | 7.893 |
| Ordered eager | 1 | 0.108 | 1.085 | 4.607 | **5.899** |
| Incompatibility lower bound only | 6 | 11.283 | 0.407 | 1.139 | 12.872 |
| Incompatibility + 1000 selected pairs | 6 | 11.222 | 0.424 | 0.947 | 12.645 |

The pair collector found a certified clique of size 6, which matches the
true optimum. It therefore avoids *all* smaller UNSAT target sizes. However,
the time saved is much less than the O(N^2) preprocessing overhead.

The 1,015 selected pairs (15 clique pairs plus 1,000 additional pairs)
lower k=6 SAT conflicts from 3,071 to 2,138. The corresponding SAT
solving time improves from 1.06s to 0.95s, but total runtime degrades
from 5.90s to 12.65s.

**Conclusion: do not make full pairwise incompatibility collection the
default.** The earlier 3,884-state result gave 5.09s vs 6.26s for ordered
eager, but that gain does not scale to 6,282 states.

Next focus: selective incompatible-pair discovery with a subquadratic
budget, incremental clique lower bounds, and a time-budgeted fallback
that abandons expensive preprocessing before it consumes the SAT budget.
This remains an opt-in experimental PR rather than a default change.
