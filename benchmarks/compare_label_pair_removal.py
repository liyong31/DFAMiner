"""Native A/B test: retain vs omit positive/reject input incompatibility clauses.

The comparison is deliberately limited to exactly ONE CNF change:
  keep -> ReducedEagerMinimiser('ordered')
  omit -> ReducedEagerMinimiser('both')
Both generate the same deterministic transition and product clauses
in the same order, with identical SAT variable numbering. The only
difference is the explicit [-nodes[p,i], -nodes[q,i]] clauses for each
positive input state p, negative input state q, and target state i.
They are logically implied by the target finality clauses, so both
formulas are equivalent, but SAT propagation and search can differ.

Each worker builds the exact same deterministic 3DFA from samples,
generates the CNF, boots a fresh CaDiCaL 1.9.5 solver, and solves it.
SAT results are independently verified by complete product traversal.
Workers are isolated subprocesses with per-run wall-clock timeouts.
Output metrics include input build, CNF construction, solver bootstrap,
SAT search, clauses, conflicts, and peak RSS.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.compare_large_3dfa import prepare
from eager_cnf_variants import ReducedEagerMinimiser
from cegar_minimiser import sdfa_cegar_minimiser
from pysat.solvers import Solver

TARGET_K = {'repo7':[6], 'repo8':[7], 'repo10':[8]}


def worker(case, k, mode):
    if mode not in ('keep','omit'):
        raise ValueError('unsupported encoding: ' + mode)
    start = perf_counter()
    machine, input_info = prepare(case)
    build_sec = perf_counter() - start
    alphabet = list(range(machine.num_letters))
    positives = len(machine.final_states)
    negatives = len(machine.reject_states)
    removed = positives * negatives * k
    engine = ReducedEagerMinimiser('ordered' if mode == 'keep' else 'both')

    begin = perf_counter()
    variables = engine.create_variables(k, alphabet, machine.num_states)
    cnf = engine.create_cnf(*variables,machine,k,alphabet,True,False)
    cnf_sec = perf_counter() - begin

    begin = perf_counter()
    solver = Solver(name='cadical195',bootstrap_with=cnf)
    bootstrap_sec = perf_counter() - begin
    try:
        begin = perf_counter()
        sat = solver.solve()
        solve_sec = perf_counter() - begin
        stats = solver.accum_stats()
        if sat:
            candidate = engine.construct_dfa_from_model(
                solver.get_model(),variables[1],k,alphabet)
            bad, _ = sdfa_cegar_minimiser._verify_product(
                machine,candidate,limit=1)
            if bad:
                raise AssertionError(('invalid SAT model',case,k,mode,bad))
    finally:
        solver.delete()

    data=dict(case=case,k=k,mode=mode,status='SAT' if sat else 'UNSAT',
              states=machine.num_states,transitions=input_info['input_edges'],
              positives=positives,negatives=negatives,
              explicit_pair_clauses=(removed if mode=='keep' else 0),
              pair_clauses_removed=(removed if mode=='omit' else 0),
              clauses=len(cnf),
              build_seconds=round(build_sec,5),
              cnf_seconds=round(cnf_sec,5),
              bootstrap_seconds=round(bootstrap_sec,5),
              sat_seconds=round(solve_sec,5),
              total_seconds=round(perf_counter()-start,5),
              conflicts=stats.get('conflicts'),
              decisions=stats.get('decisions'),
              propagations=stats.get('propagations'),
              peak_rss_mb=round(resource.getrusage(
                  resource.RUSAGE_SELF).ru_maxrss/1024,2))
    print('MEASURE '+json.dumps(data,sort_keys=True),flush=True)


def parent(args):
    rows = []
    for case in args.cases:
        targets = TARGET_K.get(case, args.sparse_targets)
        for k in targets:
            observed = {}
            for repetition in range(args.repeats):
                modes = args.modes[repetition%len(args.modes):] + args.modes[:repetition%len(args.modes)]
                for mode in modes:
                    start=perf_counter()
                    command=[sys.executable,'-u',str(Path(__file__).resolve()),
                             '--worker',case,str(k),mode]
                    try:
                        job=subprocess.run(command,cwd=ROOT,capture_output=True,
                                           text=True,timeout=args.timeout,
                                           env={**os.environ,'PYTHONHASHSEED':'0'})
                        result=[json.loads(ln[8:]) for ln in job.stdout.splitlines()
                                if ln.startswith('MEASURE ')]
                        if job.returncode or len(result)!=1:
                            row=dict(case=case,k=k,mode=mode,repeat=repetition,
                                     status='ERROR',
                                     wall_seconds=round(perf_counter()-start,4),
                                     error=(job.stderr+job.stdout)[-1600:])
                        else:
                            row=result[0]
                            row['repeat']=repetition
                            row['wall_seconds']=round(perf_counter()-start,4)
                            observed[mode]=row['status']
                    except subprocess.TimeoutExpired:
                        row=dict(case=case,k=k,mode=mode,repeat=repetition,
                                 status='TIMEOUT',wall_seconds=args.timeout)
                    rows.append(row)
                    print('RESULT '+json.dumps(row,sort_keys=True),flush=True)
                    Path(args.output).write_text(json.dumps(rows,indent=2))
            if len(set(observed.values()))>1:
                raise AssertionError(('SAT discrepancy',case,k,observed))
    print('COMPLETE '+json.dumps(dict(cases=args.cases,modes=args.modes,
         repeats=args.repeats,results=len(rows),timeouts=sum(
         x['status']=='TIMEOUT' for x in rows),errors=sum(
         x['status']=='ERROR' for x in rows))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',nargs=3,metavar=('CASE','K','MODE'))
    p.add_argument('--cases',nargs='+',default=['repo8','sparse9600','sparse19200'])
    p.add_argument('--sparse-targets',nargs='+',type=int,default=[5,6])
    p.add_argument('--modes',nargs='+',choices=['keep','omit'],default=['keep','omit'])
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--timeout',type=int,default=85)
    p.add_argument('--output',default='label_pair_removal_results.json')
    args=p.parse_args()
    if args.worker:
        worker(args.worker[0],int(args.worker[1]),args.worker[2])
    else:
        parent(args)


if __name__=='__main__':
    main()
