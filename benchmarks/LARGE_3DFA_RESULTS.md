# Native comparison on larger 3DFAs (2026-10-13)

Python 3.11, native PySAT/CaDiCaL 1.9.5, GitHub Actions.
Input states from repository ABBADINGO files and generated sparse labelled samples.
All four methods were tested in separate processes with eight-second
per-solver timeout; candidate acceptance is verified against the complete 3DFA.
All reported completed searches for k=OPT-1 returned UNSAT.

## Repository inputs (8s limit)

| Case | States | Transitions | k | Eager | CEGAR | CDCL | Rollback CDCL |
|---|---:|---:|---:|---:|---:|---:|---:|
| repo7 | 348 | 584 | 6 | 2.124s | 1.521s | TIMEOUT | 7.159s |
| repo8 | 392 | 662 | 7 | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT |
| repo10 | 465 | 798 | 8 | TIMEOUT | TIMEOUT | TIMEOUT | TIMEOUT |

## Generated sparse samples (8s limit)

| Case | States | Edges | k | Eager | CEGAR | CDCL | Rollback CDCL |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse600 | 1554 | 2139 | 5 | 1.621s | TIMEOUT | TIMEOUT | TIMEOUT |
| sparse2400 | 3884 | 6049 | 5 | 3.084s | 3.432s | 3.394s | TIMEOUT |

Peak RSS in the sparse2400 case (resident MB): eager 96.28,
CEGAR 30.24, original CDCL 24.45. Rollback timed out.
These are per-process peak resident measurements on the same Actions runner.

## Conclusions

CEGAR is fastest on repo7; eager SAT wins on both synthetic sparse cases.
Pure rollback CDCL never wins on this sample. A hard UNSAT instance with
hundreds of states can exceed an eight-second limit for all four algorithms.
Do not generalise from these examples to real parity-game separators.

The benchmark input writer intentionally avoids a separate bug in sdfa.__str__:
consecutive rejecting-state records are concatenated without newlines.

Short-budget native run:
https://github.com/liyong31/DFAMiner/actions/runs/38397358907

Benchmark driver: benchmarks/compare_large_3dfa.py

An independent longer-budget workflow run is also available in Actions;
see its completion status before drawing conclusions about timeouts.
