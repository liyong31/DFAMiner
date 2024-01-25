# DFAMiner
A python tool for inferring minimal DFA from samples.


## Requirements 

* python 3.7+
* [PySAT](https://github.com/pysathq/pysat)

## Installation

TODO

## Usage

For usage, type <b>`python3 dfaminer --help`</b>.

```
sage: dfaminer.py [-h] --file path --out path [--solver {glucose4,minisat22,maplechrono,mergesat3,gluecard4,lingeling,cadical153,cadical103,glucose42}] [--lower LOWER] [--upper UPPER] [--sdfa]
                   [--nobfs] [--safety] [--verify]

Infer minimal DFA consistent with samples

options:
  -h, --help            show this help message and exit
  --file path           path to input FA
  --out path            path to output FA
  --solver {glucose4,minisat22,maplechrono,mergesat3,gluecard4,lingeling,cadical153,cadical103,glucose42}
                        choose the SAT solver
  --lower LOWER         the lower bound for the DFA
  --upper UPPER         the upper bound for the DFA
  --sdfa                use SDFA for inference
  --nobfs               disable the constraints for BFS tree
  --safety              construct safety DFA for solving parity games
  --verify              verify resultant DFA
```

#### SAT solvers
 
For more SAT solvers, see [PySAT toolkit](https://github.com/pysathq/pysat).

