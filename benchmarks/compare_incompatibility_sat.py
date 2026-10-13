"""Native eager SAT time with incremental incompatibility and optional pair lemmas."""
import json
import resource
import sys
from pathlib import Path
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sdfa import sdfa
from strunion import dfa_builder, word_type
from eager_cnf_variants import ReducedEagerMinimiser
from incompatibility_eager import IncompatibilityEagerMinimiser
from cegar_minimiser import sdfa_cegar_minimiser
from pysat.solvers import Solver
from benchmarks.compare_large_3dfa import read_samples, make_sparse


def build(name):
    sigma,samples=read_samples(name) if name.startswith('repo') else make_sparse(name)
    b=dfa_builder(collect_incompatibility=True)
    for w,sign in sorted(samples):
        if w: b.add(w,word_type.ACCEPT if sign==1 else word_type.REJECT)
    m=dfa_builder.build(b,sigma)
    return m


def run(name,k,budget):
    start=perf_counter()
    m=build(name)
    tbuild=perf_counter()-start
    obj=ReducedEagerMinimiser('ordered') if budget==-1 else IncompatibilityEagerMinimiser(pair_budget=budget,verbose=False)
    start=perf_counter()
    alphabet=list(range(m.num_letters))
    variables=obj.create_variables(k,alphabet,m.num_states)
    cnf=obj.create_cnf(*variables,m,k,alphabet,True,False)
    tcnf=perf_counter()-start
    start=perf_counter()
    with Solver(name='cadical195',bootstrap_with=cnf) as sat:
        found=sat.solve()
        stats=sat.accum_stats()
        if found:
            dfa=obj.construct_dfa_from_model(sat.get_model(),variables[1],k,alphabet)
            violations,_=sdfa_cegar_minimiser._verify_product(m,dfa,1)
            assert not violations
    tsat=perf_counter()-start
    print('PAIR_BENCH',json.dumps(dict(case=name,k=k,budget=budget,
          clique=len(m.incompatibility.clique),states=m.num_states,
          clauses=len(cnf),build_seconds=round(tbuild,5),
          cnf_seconds=round(tcnf,5),sat_seconds=round(tsat,5),
          sat=found,conflicts=stats.get('conflicts'),
          decisions=stats.get('decisions'),rss_mb=resource.getrusage(
              resource.RUSAGE_SELF).ru_maxrss/1024)),flush=True)

if __name__=='__main__':
    name=sys.argv[1]
    k=int(sys.argv[2])
    budget=int(sys.argv[3])
    run(name,k,budget)
