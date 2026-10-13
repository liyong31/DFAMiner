"""Compare original and opt-in CEGAR minimisers on example 3DFAs.

Run from repository root:
  python benchmarks/compare_minimisers.py
  python benchmarks/compare_minimisers.py intermediate_3dfa.txt
  python benchmarks/compare_minimisers.py --upper 12 input1.txt input2.txt
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cegar_minimiser import sdfa_cegar_minimiser
from minimiser import sdfa_minimiser, solver_choices
from sdfa import sdfa


def run_one(filename, args):
    source = sdfa()
    source.load(str(filename))
    kwargs = dict(input_sdfa=source, sat=args.solver, lbound=args.lower,
                  ubound=args.upper, nobfs=args.nobfs, safety=args.safety)
    start = time.perf_counter()
    legacy = sdfa_minimiser().minimise(**kwargs)
    old_time = time.perf_counter() - start
    start = time.perf_counter()
    c = sdfa_cegar_minimiser(batch_size=args.batch, seed_per_label=args.seed, fallback_ratio=args.fallback)
    new = c.minimise(**kwargs)
    cegar_time = time.perf_counter() - start
    for name, result in [('legacy', legacy), ('CEGAR', new)]:
        if result is not None:
            errors, _ = c._verify_product(source, result, limit=1)
            if errors:
                raise AssertionError('%s misclassifies %s: %r' %
                                     (name, filename, errors[0]))
    old_size = legacy.num_states if legacy is not None else None
    new_size = new.num_states if new is not None else None
    if old_size != new_size:
        raise AssertionError('Minimum differs: %s %r vs %r' %
                             (filename, old_size, new_size))
    return dict(input=str(filename), input_states=source.num_states,
                input_transitions=source.get_num_trans(), minimum=new_size,
                legacy_seconds=round(old_time, 6),
                cegar_seconds=round(cegar_time, 6),
                cegar_iterations=sum(x['iterations'] for x in c.stats),
                cegar_counterexamples=sum(x['counterexamples'] for x in c.stats),
                cegar_clauses=sum(x['clauses'] for x in c.stats),
                cegar_stats=c.stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='*', default=['intermediate_3dfa.txt'],
                        help='textual 3DFA input files')
    parser.add_argument('--solver', choices=sorted(solver_choices), default='cadical153')
    parser.add_argument('--lower', type=int, default=1)
    parser.add_argument('--upper', type=int, default=sys.maxsize)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seed', type=int, default=4)
    parser.add_argument('--fallback', type=float, default=0.75,
                        help='fraction of active edges at which CEGAR becomes eager')
    parser.add_argument('--nobfs', action='store_true')
    parser.add_argument('--safety', action='store_true')
    parser.add_argument('--json-output', default=None)
    args = parser.parse_args()
    reports = []
    print('{:<26} {:>8} {:>8} {:>12} {:>12} {:>9}'.format(
        'Input', '3DFA', 'Minimum', 'Legacy (s)', 'CEGAR (s)', 'Speedup'))
    for input_file in args.inputs:
        path = Path(input_file)
        if not path.exists():
            path = Path(__file__).resolve().parents[1] / input_file
        record = run_one(path, args)
        reports.append(record)
        speedup = (record['legacy_seconds'] / record['cegar_seconds']
                   if record['cegar_seconds'] else float('inf'))
        print('{:<26} {:>8} {:>8} {:>12.4f} {:>12.4f} {:>8.2f}x'.format(
            path.name, record['input_states'], str(record['minimum']),
            record['legacy_seconds'], record['cegar_seconds'], speedup))
    if args.json_output:
        with open(args.json_output, 'w') as handle:
            json.dump(reports, handle, indent=2)


if __name__ == '__main__':
    main()
