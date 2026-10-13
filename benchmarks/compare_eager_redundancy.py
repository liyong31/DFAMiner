"""Compare baseline eager CNF with individual proven-redundant clause removals.

Examples: python benchmarks/compare_eager_redundancy.py --cases repo7 repo8 repo10
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pysat.solvers import Solver
from compare_large_3dfa import prepare, KNOWN_REPO_OPT
from eager_cnf_variants import ReducedEagerMinimiser
from cegar_minimiser import sdfa_cegar_minimiser
from sdfa import sdfa


def save_3dfa(machine, path):
    with path.open('w') as f:
        f.write(f'{machine.num_states} {machine.num_letters}\n')
        for p in sorted(machine.init_states): f.write(f'i {p}\n')
        for p, transitions in enumerate(machine.trans):
            for a, q in sorted(transitions.items()): f.write(f't {p} {a} {q}\n')
        for p in sorted(machine.final_states): f.write(f'a {p}\n')
        for p in sorted(machine.reject_states): f.write(f'r {p}\n')


def solve(path, k, variant, bfs):
    machine=sdfa()
    machine.load(str(path))
    alphabet=list(range(machine.num_letters))
    engine=ReducedEagerMinimiser(variant)
    start=time.perf_counter()
    nodes, edges, parents, ta, ma = engine.create_variables(k,alphabet,machine.num_states)
    clauses=engine.create_cnf(nodes,edges,parents,ta,ma,machine,k,alphabet,not bfs,False)
    generated=time.perf_counter()-start
    with Solver(name='cadical195',bootstrap_with=clauses) as solver:
        sat=solver.solve()
        stats=solver.accum_stats()
        if sat:
            dfa=engine.construct_dfa_from_model(solver.get_model(),edges,k,alphabet)
            violations,_=sdfa_cegar_minimiser._verify_product(machine,dfa,1)
            if violations: raise AssertionError((variant,k,violations[0]))
    return dict(variant=variant,k=k,sat=sat,
                seconds=round(time.perf_counter()-start,5),
                build_seconds=round(generated,5),
                clauses=len(clauses),
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,2),
                solver_stats=stats)


def worker(path,k,variant,bfs,timeout,mem):
    command=[sys.executable,'-u',__file__,'--single',str(path),
             '--k',str(k),'--variant',variant]
    if bfs: command.append('--bfs')
    def limit():
        if mem:
            resource.setrlimit(resource.RLIMIT_AS,(mem*1024*1024,mem*1024*1024))
    start=time.perf_counter()
    try:
        p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,
                         timeout=timeout,preexec_fn=limit,
                         env={**os.environ,'PYTHONHASHSEED':'0'})
        records=[x[len('MEASURE '):] for x in p.stdout.splitlines()
                 if x.startswith('MEASURE ')]
        if p.returncode or len(records)!=1:
            return dict(variant=variant,k=k,status='ERROR',
                        seconds=round(time.perf_counter()-start,3),
                        error=(p.stderr+p.stdout)[-1500:])
        data=json.loads(records[0])
        data['status']='SAT' if data['sat'] else 'UNSAT'
        return data
    except subprocess.TimeoutExpired:
        return dict(variant=variant,k=k,status='TIMEOUT',seconds=timeout)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--cases',nargs='+',default=['repo7','repo8','repo10','sparse600','sparse2400'])
    p.add_argument('--variants',nargs='+',default=['baseline','ordered','implied','both','dedup'])
    p.add_argument('--timeout',type=int,default=16)
    p.add_argument('--memory-mb',type=int,default=3800)
    p.add_argument('--bfs',action='store_true')
    p.add_argument('--output',default='eager_redundancy.json')
    p.add_argument('--single',type=Path)
    p.add_argument('--k',type=int)
    p.add_argument('--variant')
    args=p.parse_args()
    if args.single:
        print('MEASURE '+json.dumps(solve(args.single,args.k,args.variant,args.bfs)),flush=True)
        return
    rows=[]
    with tempfile.TemporaryDirectory() as dirname:
        for case in args.cases:
            machine,info=prepare(case)
            file=Path(dirname)/(case+'.txt')
            save_3dfa(machine,file)
            k=KNOWN_REPO_OPT[case]-1 if case.startswith('repo') else 5
            print('CASE '+json.dumps({**info,'k':k}),flush=True)
            outcomes=set()
            for variant in args.variants:
                data=worker(file,k,variant,args.bfs,args.timeout,args.memory_mb)
                data.update(case=case,input_states=machine.num_states)
                rows.append(data)
                print('RESULT '+json.dumps(data),flush=True)
                if data['status'] in ('SAT','UNSAT'): outcomes.add(data['status'])
            if len(outcomes)>1:raise AssertionError((case,outcomes))
            Path(args.output).write_text(json.dumps(rows,indent=2))
    print('COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    main()
