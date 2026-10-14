"""Native CaDiCaL benchmark of cheap structural information for eager SAT.

Every (case, mode) runs in a separate subprocess.  Timeouts, errors and
partially completed suites are reported without hiding completed results.

Examples:
  python benchmarks/compare_scored_information.py --seconds 30
  python benchmarks/compare_scored_information.py --cases repo8 repo10 --seconds 60
  python benchmarks/compare_scored_information.py --cases sparse2400 sparse4800 --repeat 3
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

from benchmarks.compare_large_3dfa import prepare, make_sparse, build_3dfa
from cheap_lower_bound import CheapIncompatibilityBound
from cheap_bound_eager import CheapBoundEagerMinimiser
from eager_cnf_variants import ReducedEagerMinimiser
from cegar_minimiser import sdfa_cegar_minimiser
from pysat.solvers import Solver

TARGETS = {'repo7':6, 'repo8':7, 'repo10':8, 'sparse600':5,
           'sparse2400':5, 'sparse4800':5, 'sparse9600':5, 'sparse19200':5}
MODES = {'baseline':('baseline',0), 'bound':('impact',0),
         'impact100':('impact',100), 'impact300':('impact',300),
         'impact600':('impact',600), 'short300':('short',300),
         'degree300':('degree',300), 'fanin300':('fanin',300)}


def prepare_machine(case):
    if case in ('sparse4800','sparse9600','sparse19200'):
        return build_3dfa(*make_sparse(case))
    return prepare(case)[0]


def worker(case, mode):
    start = perf_counter()
    machine = prepare_machine(case)
    build_seconds = perf_counter() - start
    k = TARGETS[case]
    policy, budget = MODES[mode]
    start_pre = perf_counter()
    info = (CheapIncompatibilityBound(
        machine, budget_ms=25, max_candidates=256,
        max_suffixes=64, max_depth=9)
        if mode != 'baseline' else None)
    pairs = (info.scored_pairs(limit=budget,budget_ms=15,policy=policy)
             if info and budget else [])
    if info:
        info.verify()
        info.verify_scored_pairs(pairs)
    pre_seconds = perf_counter() - start_pre
    engine = (CheapBoundEagerMinimiser(verbose=False)
              if info is not None else ReducedEagerMinimiser('ordered'))
    engine.selected_pairs = pairs
    alphabet = list(range(machine.num_letters))
    start_cnf = perf_counter()
    variables = engine.create_variables(k, alphabet, machine.num_states)
    clauses = engine.create_cnf(*variables, machine, k, alphabet, True, False)
    cnf_seconds = perf_counter() - start_cnf
    start_sat = perf_counter()
    with Solver(name='cadical195', bootstrap_with=clauses) as solver:
        sat = solver.solve()
        stats = solver.accum_stats()
        if sat:
            candidate = engine.construct_dfa_from_model(
                solver.get_model(),variables[1],k,alphabet)
            bad,_ = sdfa_cegar_minimiser._verify_product(
                machine,candidate,limit=1)
            assert not bad, (case,mode,k,bad)
    sat_seconds = perf_counter() - start_sat
    data = dict(case=case,mode=mode,k=k,status='SAT' if sat else 'UNSAT',
                states=machine.num_states,transitions=sum(map(len,machine.trans)),
                bound=info.lower_bound if info else None,
                suffixes=len(info.suffixes) if info else 0,
                selected_pairs=len(pairs),clauses=len(clauses),
                input_build_seconds=round(build_seconds,5),
                preprocessing_seconds=round(pre_seconds,5),
                cnf_seconds=round(cnf_seconds,5),
                sat_seconds=round(sat_seconds,5),
                total_seconds=round(perf_counter()-start,5),
                conflicts=stats.get('conflicts'),
                decisions=stats.get('decisions'),
                propagations=stats.get('propagations'),
                peak_rss_mb=round(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss/1024,2))
    print('MEASURE '+json.dumps(data,sort_keys=True),flush=True)


def run_suite(args):
    rows=[]
    for case in args.cases:
        for repeat in range(args.repeat):
            observations={}
            for mode in args.modes:
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),
                     '--worker',case,mode]
                start=perf_counter()
                try:
                    p=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,text=True,
                                     timeout=args.seconds,
                                     env={**os.environ,'PYTHONHASHSEED':'0'})
                    metrics=[json.loads(line.split('MEASURE ',1)[1])
                             for line in p.stdout.splitlines()
                             if line.startswith('MEASURE ')]
                    if p.returncode or len(metrics)!=1:
                        row={'case':case,'mode':mode,'repeat':repeat,
                             'status':'ERROR',
                             'seconds':round(perf_counter()-start,3),
                             'error':(p.stderr + p.stdout)[-1200:]}
                    else:
                        row=metrics[0]
                        row['repeat']=repeat
                        observations[mode]=row['status']
                except subprocess.TimeoutExpired:
                    row={'case':case,'mode':mode,'repeat':repeat,
                         'status':'TIMEOUT','seconds':args.seconds}
                rows.append(row)
                print('RESULT '+json.dumps(row,sort_keys=True),flush=True)
                Path(args.output).write_text(json.dumps(rows,indent=2))
            if len(set(observations.values()))>1:
                raise AssertionError(('SAT disagreement',case,repeat,observations))
    print('COMPLETE '+json.dumps(
        {'measurements':len(rows),
         'timeout_or_error':sum(x['status'] in ('TIMEOUT','ERROR') for x in rows),
         'cases':args.cases,'modes':args.modes}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',nargs=2,metavar=('CASE','MODE'))
    p.add_argument('--cases',nargs='+',choices=tuple(TARGETS),
                   default=['repo7','repo8','repo10','sparse2400','sparse4800'])
    p.add_argument('--modes',nargs='+',choices=tuple(MODES),
                   default=['baseline','bound','impact100',
                            'impact300','short300','degree300'])
    p.add_argument('--seconds',type=int,default=30)
    p.add_argument('--repeat',type=int,default=1)
    p.add_argument('--output',default='scored_eager_results.json')
    args=p.parse_args()
    if args.worker:
        worker(*args.worker)
    else:
        run_suite(args)
