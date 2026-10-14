"""Native CaDiCaL A/B: compact eager BFS symmetry with scored pair lemmas.

Each configuration is a separate process with a hard wall-clock timeout.
The same input samples and CNF variable numbering are used across variants.

Benchmark k=OPT-1 UNSAT and k=OPT SAT (where known).
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmarks.compare_large_3dfa import prepare
from hybrid_eager import HybridEagerMinimiser
from cegar_minimiser import sdfa_cegar_minimiser
from pysat.solvers import Solver

CASES={'repo7':(6,7),'repo8':(7,8),'repo10':(8,9),
       'sparse2400':(5,6),'sparse4800':(5,6),
       'sparse9600':(5,6),'sparse19200':(5,6)}
MODES={
 'compact-bfs':(0,'short',False,True),
 'short100':(100,'short',False,True),
 'short300':(300,'short',False,True),
 'degree300':(300,'degree',False,True),
 'impact300':(300,'impact',False,True),
 'short600':(600,'short',False,True),
 'explicit-bfs':(0,'short',True,True),
 'compact-no-bfs':(0,'short',False,False),
}

def worker(case,k,mode):
    start=perf_counter()
    m,info=prepare(case)
    built=perf_counter()-start
    n_pairs,policy,explicit,bfs=MODES[mode]
    engine=HybridEagerMinimiser(
        pair_budget=n_pairs,policy=policy,
        bound_budget_ms=25 if n_pairs else 0,
        pair_budget_ms=15,explicit_label_pairs=explicit)
    engine.prepare(m)
    prep=perf_counter()-start-built
    alpha=list(range(m.num_letters))
    begin=perf_counter()
    variables=engine.create_variables(k,alpha,m.num_states)
    clauses=engine.create_cnf(*variables,m,k,alpha,not bfs,False)
    cnf_time=perf_counter()-begin
    begin=perf_counter()
    with Solver(name='cadical195',bootstrap_with=clauses) as solver:
        bootstrap_time=perf_counter()-begin
        begin=perf_counter()
        result=solver.solve()
        search_time=perf_counter()-begin
        stats=solver.accum_stats()
        if result:
            candidate=engine.construct_dfa_from_model(
                solver.get_model(),variables[1],k,alpha)
            violations,_=sdfa_cegar_minimiser._verify_product(m,candidate,1)
            assert not violations,(case,mode,k,violations)
    print('MEASURE '+json.dumps(dict(
        case=case,k=k,mode=mode,states=m.num_states,
        input_build_seconds=round(built,5),
        prep_seconds=round(prep,5),
        clique_lb=engine.lower_bound,
        selected_pairs=len(engine.selected_pairs),
        cnf_seconds=round(cnf_time,5),
        bootstrap_seconds=round(bootstrap_time,5),
        sat_seconds=round(search_time,5),
        total_seconds=round(perf_counter()-start,5),
        clauses=len(clauses),
        status='SAT' if result else 'UNSAT',
        conflicts=stats.get('conflicts'),decisions=stats.get('decisions')
    ),sort_keys=True),flush=True)

def parent(args):
    out=[]
    for case in args.cases:
        targets=args.targets or CASES[case]
        for k in targets:
            results={}
            for mode in args.modes:
                start=perf_counter()
                try:
                    job=subprocess.run(
                        [sys.executable,'-u',__file__,'--worker',case,str(k),mode],
                        cwd=ROOT,capture_output=True,text=True,
                        timeout=args.timeout,
                        env={**os.environ,'PYTHONHASHSEED':'0'})
                    parsed=[json.loads(s[8:]) for s in job.stdout.splitlines()
                            if s.startswith('MEASURE ')]
                    if job.returncode or len(parsed)!=1:
                        row=dict(case=case,k=k,mode=mode,status='ERROR',
                                 error=(job.stderr+job.stdout)[-1800:])
                    else:
                        row=parsed[0]
                        results[mode]=row['status']
                except subprocess.TimeoutExpired:
                    row=dict(case=case,k=k,mode=mode,status='TIMEOUT',
                             total_seconds=args.timeout)
                out.append(row)
                print('RESULT '+json.dumps(row,sort_keys=True),flush=True)
                Path(args.output).write_text(json.dumps(out,indent=2))
            if len(set(results.values()))>1:
                raise AssertionError((case,k,results))
    print('COMPLETE '+str(len(out)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',nargs=3)
    parser.add_argument('--cases',nargs='+',default=['repo7','repo8','repo10','sparse2400','sparse9600'])
    parser.add_argument('--targets',nargs='+',type=int)
    parser.add_argument('--modes',nargs='+',choices=list(MODES),
                        default=['compact-bfs','short100','short300','degree300','impact300','short600'])
    parser.add_argument('--timeout',type=int,default=30)
    parser.add_argument('--output',default='hybrid_bfs_scores.json')
    args=parser.parse_args()
    if args.worker:worker(args.worker[0],int(args.worker[1]),args.worker[2])
    else:parent(args)
