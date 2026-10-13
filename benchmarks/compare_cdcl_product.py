"""Exact side-by-side comparison of eager, CEGAR and CDCL-product modes.

Usage: python benchmarks/compare_cdcl_product.py intermediate_3dfa.txt
Requires native python-sat with CaDiCaL 1.9.5. All returned candidates are
independently verified against the full input three-valued automaton.
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdfa import sdfa
from minimiser import sdfa_minimiser
from cegar_minimiser import sdfa_cegar_minimiser
from cdcl_product import CDCLProductMinimiser


def run(path, limit, nobfs, safety):
    source = sdfa()
    source.load(str(path))
    runners = (
        ('eager', sdfa_minimiser(), 'cadical195'),
        ('cegar', sdfa_cegar_minimiser(verbose=False), 'cadical195'),
        ('cdcl_product', CDCLProductMinimiser(verbose=False), 'cadical195'),
    )
    outcomes = []
    for name, runner, solver in runners:
        before = perf_counter()
        automaton = runner.minimise(source, solver, 1, limit, nobfs, safety)
        elapsed = round(perf_counter() - before, 4)
        size = None if automaton is None else automaton.num_states
        if automaton is not None:
            errors, _ = sdfa_cegar_minimiser._verify_product(source, automaton, 1)
            if errors:
                raise AssertionError((name, str(path), errors[0]))
        outcomes.append(dict(input=str(path), mode=name, size=size,
                             seconds=elapsed,
                             stats=getattr(runner, 'stats', None)))
        print('{} {}: size={} elapsed={}s'.format(path, name, size, elapsed),
              flush=True)
    if len({result['size'] for result in outcomes}) != 1:
        raise AssertionError('minimum DFA size differs between solvers')
    return outcomes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('inputs', nargs='*', default=['intermediate_3dfa.txt'])
    p.add_argument('--upper', type=int, default=sys.maxsize)
    p.add_argument('--nobfs', action='store_true')
    p.add_argument('--safety', action='store_true')
    p.add_argument('--json-output')
    args = p.parse_args()
    output = []
    for path in args.inputs:
        if not Path(path).exists():
            path = str(Path(__file__).resolve().parents[1] / path)
        output.extend(run(path, args.upper, args.nobfs, args.safety))
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
