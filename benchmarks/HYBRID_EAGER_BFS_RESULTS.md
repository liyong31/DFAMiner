# Compact eager + target BFS + scored incompatibility clauses

Native CI: https://github.com/liyong31/DFAMiner/actions/runs/38565706883
Date: 14 October 2026. Source branch: codex/hybrid-eager-bfs-scored,
based on cegar (209f97e). The baseline is the same *compact* eager CNF
(mode='both': no quadratic positive/reject clauses) and **existing target BFS
symmetry**. Additional variants use a cheap 25ms suffix collector plus up
to 15ms pair scoring, then add witnessed redundant `(-x[p,i] or -x[q,i])`
clauses. All numbers are isolated single native CaDiCaL 1.9.5 processes,
including input generation, preprocessing, CNF generation and SAT. All
SAT candidates passed independent full-product verification.

| Case | k | Result | Compact BFS | Degree300 | Short300 | Short600 | Impact300 |
|---|---:|---|---:|---:|---:|---:|---:|
| repo7, 348 states | 6 | UNSAT | **0.0505** | 0.0691 | 0.0762 | 0.0742 | 0.0749 |
| repo7, 348 states | 7 | SAT | 0.1152 | **0.0801** | 0.0831 | 0.1148 | 0.1376 |
| repo8, 392 states | 7 | UNSAT | 0.1904 | **0.0928** | 0.1295 | 0.1140 | 0.1378 |
| repo8, 392 states | 8 | SAT | 0.3350 | **0.1021** | 0.1853 | 0.2082 | 0.3998 |
| repo10, 465 states | 8 | UNSAT | 0.5001 | **0.1304** | 0.4907 | 0.4801 | 0.4512 |
| repo10, 465 states | 9 | SAT | 1.1998 | **0.3711** | 1.1262 | 0.6051 | 0.5927 |
| sparse2400, 3,884 states | 5 | UNSAT | 0.4536 | **0.4117** | 0.4494 | 0.4199 | 0.4249 |
| sparse2400, 3,884 states | 6 | SAT | **0.3137** | 0.4656 | 0.5083 | 0.4968 | 0.4735 |
| sparse4800, 6,282 states | 5 | UNSAT | 1.0142 | 1.0135 | 0.9062 | **0.8434** | 0.9109 |
| sparse4800, 6,282 states | 6 | SAT | 0.9097 | 1.1032 | **0.7703** | 0.8154 | 3.5964 |
| sparse9600, 10,107 states | 5 | UNSAT | 1.9150 | 2.0009 | 2.0491 | 1.9308 | **1.8424** |
| sparse9600, 10,107 states | 6 | SAT | **1.7548** | 1.9294 | 6.6921 | 3.0339 | 1.9464 |

**Interpretation.** Scored pair clauses are *complementary* to target
BFS on the harder medium examples: `degree300` gives 2.05x to 3.84x
speedups at k=7/8 UNSAT and k=8/9 SAT. On large sample families where
BFS already makes SAT cheap, the effect is mixed; `short300` at 10,107
states k=6 notably regresses 1.75s to 6.69s.

Do **not** globally enable scored pair clauses with target BFS. Use
compact BFS as a strong default baseline; consider heuristics or SAT
portfolio selectively. Investigate repeatability and clause-order effects.
Small-case lower bounds can eliminate smaller k searches; this table
deliberately compares *fixed* k and does not include that benefit.

The module hybrid_eager.py supports `HybridEagerMinimiser.prepare()` and
`minimise()` but is experimental. It has not been wired to the default
CLI and is not merged into cegar.
