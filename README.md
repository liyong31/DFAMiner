# DFAMiner
A python tool for mining a minimal DFA from a given set of labelled samples.

For details of the algorithm, please read the paper [[1]](#1).



## Requirements 

* Python 3.7+
* [DFA](https://github.com/mvcisback/dfa)
* [PySAT](https://github.com/pysathq/pysat)

## Installation

DFAMiner requires additional Python libraries that are usually not available in the ordinary software repository of the used operating system.
To avoid creating inconsistencies with the Python libraries installed system-wide, DFAMiner can be used inside a virtual environment.
This can be achieved as follows:
* Download and install a virtual environment manager; below we refer to [MiniConda](https://www.anaconda.com/docs/getting-started/miniconda/main)
* Create a virtual environment
```
conda create -n dfaminer python=3.10
conda activate dfaminer
```
* Install the required packages
```
pip install dfa python-sat[aiger,approxmc,cryptosat,pblib]
```
* To deactivate the environment, just use
```
conda deactivate
```

## Usage

For usage, type <b>`python3 dfaminer.py --help`</b>.

```
usage: dfaminer.py [-h] --file path --out path [--output-format {dot,textual}] [--intermediate path]
                   [--solver {glucose4,glucose42,minisat22,cadical103,cadical153,lingeling,maplechrono,gluecard4,cadical195,mergesat3}] [--lower LOWER] [--upper UPPER] [--3dfa] [--nobfs]
                   [--safety] [--verify]

Mining a minimal DFA consistent with samples

options:
  -h, --help            show this help message and exit
  --file path           path to input sample file
  --out path            path to output DFA
  --output-format {dot,textual}
                        the format for the output
  --intermediate path   path to output the intermediate 3DFA
  --solver {glucose4,glucose42,minisat22,cadical103,cadical153,lingeling,maplechrono,gluecard4,cadical195,mergesat3}
                        choose the SAT solver
  --lower LOWER         the lower bound for the DFA
  --upper UPPER         the upper bound for the DFA
  --3dfa                use three valued DFA for inference
  --nobfs               disable the constraints for BFS tree
  --safety              construct safety DFA for solving parity games
  --verify              verify resultant DFA
```

#### SAT solvers
 
For more SAT solvers, see [PySAT toolkit](https://github.com/pysathq/pysat).

#### Input format

dfaminer.py accepts a file in [Abbadingo format](https://abbadingo.cs.nuim.ie/).
An example file is given below:
```
8 2
1 3 0 0 0
1 3 0 0 1
0 3 0 1 0
0 3 0 1 1
1 3 1 0 0
0 3 1 0 1
0 3 1 1 0
0 3 1 1 1
```
The first line gives the number of samples and the size of the alphabet.
Each line after that will first specify the membership of the word, the length of the word and the word sequence.
Here `1` means accept, `0` reject and `-1` don't care.

It also accepts an equivalent JSON file as follows:
```
{ 
  "alphabet": ["0", "1"], 
  "accepting": ["000", "001", "100"], 
  "rejecting": ["010", "011", "101", "110", "111"]
}
where "accepting" and "rejecting" are mandatory arrays providing the positive and negative samples, respectively, given as strings. 
 "alphabet", if present, gives the alphabet as a vector; if it is absent, then the alphabet is derived from the symbols occurring in the given samples.
```
The positive and negative samples can be directly provided as Python variables, by means of the following Python code:
```
positive_samples = [['0', '0', '0'], ['0', '0', '1'], ['1', '0', '0']]
negative_samples = [['0', '1', '0'], ['0', '1', '1'], ['1', '0', '1'], ['1', '1', '0'], ['1', '1', '1']]
``` 

#### Minimiser
One can also use minimiser.py as a standalone tool to minimise DFAs with don't care words.
An example input file is `intermediate_3dfa.txt`:
```
8 2
i 0
t 0 0 2
t 0 1 6
t 2 0 3
t 2 1 4
t 3 0 1
t 3 1 1
t 4 0 5
t 4 1 5
t 6 0 7
t 6 1 4
t 7 0 1
t 7 1 5
a 1
r 5
```


The format is quite straight-forward. 
The first line gives the number of states and the size of the alphabet.
For each line, `i` is followed by an initial state, `a` by a final state, `r` by a reject state, and `t` by a transition (source, letter, destination)

#### License
See LICENSE.txt


## References
<a id="1">[1]</a> 
Daniele Dell’Erba, Yong Li & Sven Schewe (2024). 
DFAMiner: Mining Minimal Separating DFAs from Labelled Samples. 
26th International Symposium on Formal Methods (FM 2024), LNCS 14934, 48--66.