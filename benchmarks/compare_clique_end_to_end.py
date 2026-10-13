"""One-minute native, end-to-end lower-bound benefit experiment.

Compare the *same* sparse2400 (3,884-state) 3DFA with four configurations:
 - eager-default: original eager clauses, search k=1..6
 - eager-ordered: duplicate-free transition pairs, search k=1..6
 - clique-bound: collect all pairs and compute clique, ordered eager k=clique..6,
                 but add NO incompatibility clauses (isolates lower-bound benefit)
 - clique-pairs1000: same lower bound plus 1000 selected incompatible pairs

Each configuration runs in its own process with a 60-second wall-clock limit,
including all input generation, construction, all SAT calls and verification.
Results include separate build / CNF / SAT time and status for each tested k.
No solver or CNF reuse between target sizes (matches existing minimisation).
"""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.compare_large_3dfa import make_sparse
from strunion import dfa_builder, word_type
from minimiser import sdfa_minimiser
from eager_cnf_variants import ReducedEagerMinimiser
from incompatibility_eager import IncompatibilityEagerMinimiser
from cegar_minimiser import sdfa_cegar_minimiser
from pysat.solvers import Solver

MODES = ('eager-default', 'eager-ordered', 'clique-bound', 'clique-pairs1000')


def build_input(mode):
    start = perf_counter()
    alphabet, samples = make_sparse('sparse2400')
    need_pairs = mode.startswith('clique-')
    builder = dfa_builder(collect_incompatibility=need_pairs)
    for word, label in sorted(samples):
        builder.add(word, word_type.ACCEPT if label == 1 else word_type.REJECT)
    machine = dfa_builder.build(builder, alphabet)
    info = getattr(machine, 'incompatibility', None)
    if info:
        info.verify()
    assert len(samples) == 2400
    assert machine.num_states == 3884, machine.num_states
    return machine, dict(
        build_seconds=round(perf_counter() - start, 5),
        input_states=machine.num_states,
        input_edges=sum(map(len, machine.trans)),
        samples=len(samples),
        certified_clique_size=(len(info.clique) if info else None),
        selected_pairs=(len(info.selected_pairs(1000)) if mode == 'clique-pairs1000'
                        else None),
    )


def solve_size(machine, size, mode):
    alphabet = list(range(machine.num_letters))
    engine = (sdfa_minimiser() if mode == 'eager-default'
              else IncompatibilityEagerMinimiser(mode='ordered',
                   pair_budget=1000, verbose=False)
                   if mode == 'clique-pairs1000'
              else ReducedEagerMinimiser(mode='ordered'))
    start = perf_counter()
    variables = engine.create_variables(size, alphabet, machine.num_states)
    clauses = engine.create_cnf(
        *variables, machine, size, alphabet, True, False)
    cnf_seconds = perf_counter() - start
    start = perf_counter()
    with Solver(name='cadical195', bootstrap_with=clauses) as solver:
        satisfiable = solver.solve()
        stats = solver.accum_stats()
        candidate = (engine.construct_dfa_from_model(
            solver.get_model(), variables[1], size, alphabet)
                     if satisfiable else None)
    sat_seconds = perf_counter() - start
    if candidate:
        mismatches, _ = sdfa_cegar_minimiser._verify_product(
            machine, candidate, limit=1)
        assert not mismatches, (mode, size, mismatches)
    return dict(k=size, status='SAT' if satisfiable else 'UNSAT',
                clauses=len(clauses), cnf_seconds=round(cnf_seconds, 5),
                sat_seconds=round(sat_seconds, 5),
                conflicts=stats.get('conflicts'),
                decisions=stats.get('decisions'),
                propagations=stats.get('propagations'))


def run_mode(mode):
    begin = perf_counter()
    machine, input_record = build_input(mode)
    lower = (input_record['certified_clique_size']
             if mode.startswith('clique-') else 1)
    print('BUILT ' + json.dumps({'mode': mode, **input_record,
                                  'starting_k': lower}), flush=True)
    proof = []
    for k in range(lower, 7):
        record = solve_size(machine, k, mode)
        proof.append(record)
        print('K_RESULT ' + json.dumps({'mode': mode, **record}), flush=True)
        if record['status'] == 'SAT':
            break
    status = ('OPTIMUM_CERTIFIED' if proof and proof[-1]['status'] == 'SAT'
              else 'NO_SOLUTION_WITHIN_BOUND')
    answer = dict(mode=mode, status=status,
                  total_seconds=round(perf_counter() - begin, 5),
                  **input_record, lower_bound=lower,
                  attempted_sizes=[item['k'] for item in proof],
                  sat_seconds_total=round(sum(x['sat_seconds'] for x in proof),5),
                  cnf_seconds_total=round(sum(x['cnf_seconds'] for x in proof),5),
                  peak_rss_mb=round(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2))
    print('SUMMARY ' + json.dumps(answer), flush=True)


def run_harness(seconds, output):
    results = []
    for mode in MODES:
        start=perf_counter()
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
                   '--child', mode]
        try:
            proc = subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True,
                timeout=seconds, env={**os.environ, 'PYTHONHASHSEED': '0'})
            stdout = proc.stdout
            stderr = proc.stderr
            timed_out = False
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or b''
            stderr = exc.stderr or b''
            if isinstance(stdout, bytes): stdout = stdout.decode('utf-8','replace')
            if isinstance(stderr, bytes): stderr = stderr.decode('utf-8','replace')
            timed_out = True
            proc = None

        partial = []
        summary = None
        for line in stdout.splitlines():
            if line.startswith('K_RESULT '):
                partial.append(json.loads(line[9:]))
            elif line.startswith('SUMMARY '):
                summary = json.loads(line[8:])
            elif line.startswith('BUILT '):
                print(line,flush=True)
        for line in stdout.splitlines():
            if line.startswith('K_RESULT '):
                print(line,flush=True)
        row = (summary if summary else dict(
            mode=mode,
            status=('TIMEOUT' if timed_out else 'ERROR'),
            total_seconds=round(perf_counter()-start,5),
            completed_k=partial,
            error=str(stderr)[-3000:]))
        results.append(row)
        print('RESULT ' + json.dumps(row, sort_keys=True),flush=True)
        Path(output).write_text(json.dumps(results, indent=2))

    completed = [r for r in results if r.get('status') == 'OPTIMUM_CERTIFIED']
    if len(completed) >= 2:
        sizes = set()
        for r in completed:
            if r['mode'].startswith('clique-'):
                # Each clique certifies all sizes below the first attempt.
                assert r['lower_bound'] == 5
            sizes.add(6)
        assert len(sizes) == 1
    print('COMPLETE ' + json.dumps({
        'configurations':len(results),
        'completed':len(completed),
        'time_limit_per_configuration_seconds':seconds,
    }), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',choices=MODES)
    parser.add_argument('--seconds',type=int,default=60)
    parser.add_argument('--output',default='clique_end_to_end_results.json')
    args=parser.parse_args()
    if args.child:
        run_mode(args.child)
    else:
        run_harness(args.seconds,args.output)


if __name__=='__main__':
    main()
