"""Native A/B comparison: ordered eager, cheap bound, cheap bound plus scored pairs."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.compare_large_3dfa import prepare,make_sparse,build_3dfa
from cheap_lower_bound import CheapIncompatibilityBound
from cheap_bound_eager import CheapBoundEagerMinimiser
from eager_cnf_variants import ReducedEagerMinimiser
from cegar_minimiser import sdfa_cegar_minimiser
from pysat.solvers import Solver

def run(m,k,policy,budget):
    t=time.perf_counter()
    bound=CheapIncompatibilityBound(m,budget_ms=25,max_candidates=256,max_suffixes=64,max_depth=9) if policy!='baseline' else None
    pairs=bound.scored_pairs(limit=budget,budget_ms=15,policy=policy) if bound and budget else []
    if bound:bound.verify();bound.verify_scored_pairs(pairs)
    collection=time.perf_counter()-t
    obj=CheapBoundEagerMinimiser(verbose=False) if bound else ReducedEagerMinimiser('ordered')
    obj.selected_pairs=pairs if bound else []
    alphabet=list(range(m.num_letters))
    v=obj.create_variables(k,alphabet,m.num_states)
    cnf=obj.create_cnf(*v,m,k,alphabet,True,False)
    built=time.perf_counter()-t
    with Solver(name='cadical195',bootstrap_with=cnf) as sol:
        sat=sol.solve();stats=sol.accum_stats()
        if sat:
            candidate=obj.construct_dfa_from_model(sol.get_model(),v[1],k,alphabet)
            assert not sdfa_cegar_minimiser._verify_product(m,candidate,limit=1)[0]
    elapsed=time.perf_counter()-t
    print('MEASURE '+json.dumps(dict(policy=policy,pairs=len(pairs),k=k,sat=sat,
        lower=bound.lower_bound if bound else 1,pre_seconds=round(collection,5),
        cnf_seconds=round(built-collection,5),
        total_seconds=round(elapsed,5),conflicts=stats.get('conflicts'),
        clauses=len(cnf))),flush=True)

for case,k in [('repo7',6),('repo8',7),('repo10',8),('sparse2400',5),('sparse4800',5)]:
    m=build_3dfa(*make_sparse(case)) if case=='sparse4800' else prepare(case)[0]
    print('CASE',case,m.num_states,flush=True)
    for policy,budget in [('baseline',0),('impact',0),('impact',100),('impact',300),('short',300),('degree',300)]:
        run(m,k,policy,budget)
