"""Bounded native benchmarks for 3DFA trimming and input-order RGS symmetry."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from time import perf_counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sdfa import sdfa
from benchmarks.compare_large_3dfa import prepare
from preprocessed_eager import PreprocessedEagerMinimiser
from cegar_minimiser import sdfa_cegar_minimiser
from pysat.solvers import Solver

def work(case,k,variant):
    t0=perf_counter()
    m,_=prepare(case)
    opts={
        'base':('both',False,'none',0),
        'trim':('both',True,'none',0),
        'rgs-bfs':('both',False,'bfs',32),
        'rgs-fanin':('both',False,'fanin',32),
        'trim-rgs':('both',True,'bfs',32),
        'target-bfs':('both',False,'none',0),
        'ordered':('ordered',False,'none',0)
    }
    mode,trim,symm,prefix=opts[variant]
    eng=PreprocessedEagerMinimiser(mode,trim,symm,prefix)
    machine,preinfo=eng.preprocess(m)
    tpre=perf_counter()-t0
    alpha=list(range(machine.num_letters))
    variables=eng.create_variables(k,alpha,machine.num_states)
    clauses=eng.create_cnf(*variables,machine,k,alpha,variant!='target-bfs',False)
    tcnf=perf_counter()-t0-tpre
    start=perf_counter()
    with Solver(name='cadical195',bootstrap_with=clauses) as sol:
        sat=sol.solve()
        stats=sol.accum_stats()
        if sat:
            dfa=eng.construct_dfa_from_model(sol.get_model(),variables[1],k,alpha)
            assert not sdfa_cegar_minimiser._verify_product(m,dfa,limit=1)[0]
    print('MEASURE '+json.dumps(dict(case=case,k=k,variant=variant,
        status='SAT' if sat else 'UNSAT',states_before=m.num_states,
        states_after=machine.num_states,removed=preinfo['removed_states'],
        clauses=len(clauses),prep_seconds=round(tpre,5),
        cnf_seconds=round(tcnf,5),sat_seconds=round(perf_counter()-start,5),
        total_seconds=round(perf_counter()-t0,5),
        conflicts=stats.get('conflicts'))),flush=True)

def parent(args):
    rows=[]
    for case in args.cases:
        k=({'repo7':6,'repo8':7,'repo10':8}.get(case,5))
        observations={}
        for variant in args.variants:
            started=perf_counter()
            try:
                proc=subprocess.run([sys.executable,'-u',__file__,'--worker',case,str(k),variant],
                    cwd=ROOT,capture_output=True,text=True,timeout=args.timeout,
                    env={**os.environ,'PYTHONHASHSEED':'0'})
                results=[json.loads(l[8:]) for l in proc.stdout.splitlines() if l.startswith('MEASURE ')]
                if proc.returncode or len(results)!=1:
                    row=dict(case=case,k=k,variant=variant,status='ERROR',
                             error=(proc.stderr+proc.stdout)[-1500:])
                else:
                    row=results[0]
                    observations[variant]=row['status']
            except subprocess.TimeoutExpired:
                row=dict(case=case,k=k,variant=variant,status='TIMEOUT',
                         total_seconds=args.timeout)
            rows.append(row)
            print('RESULT '+json.dumps(row,sort_keys=True),flush=True)
            Path(args.output).write_text(json.dumps(rows,indent=2))
        if len(set(observations.values()))>1:
            raise AssertionError((case,observations))
    print('COMPLETE '+str(len(rows)),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--worker',nargs=3)
    a.add_argument('--cases',nargs='+',default=['repo7','repo8','sparse2400','sparse9600'])
    a.add_argument('--variants',nargs='+',default=['base','trim','rgs-bfs','rgs-fanin','trim-rgs','target-bfs'])
    a.add_argument('--timeout',type=int,default=22)
    a.add_argument('--output',default='preprocess_symmetry_results.json')
    args=a.parse_args()
    if args.worker:work(args.worker[0],int(args.worker[1]),args.worker[2])
    else:parent(args)
