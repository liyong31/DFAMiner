"""End-to-end comparison: ordered eager versus bounded-suffix lower bound."""
import json
import sys
from pathlib import Path
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.compare_large_3dfa import prepare,make_sparse,build_3dfa
from cheap_lower_bound import CheapIncompatibilityBound
from eager_cnf_variants import ReducedEagerMinimiser
from pysat.solvers import Solver
from cegar_minimiser import sdfa_cegar_minimiser


def run(m, use_bound):
    t0=perf_counter()
    bound=CheapIncompatibilityBound(m,budget_ms=25,max_suffixes=64,
                                   max_candidates=256,max_depth=9) if use_bound else None
    if bound: bound.verify()
    starting=bound.lower_bound if bound else 1
    info=[]
    for k in range(starting,7):
        obj=ReducedEagerMinimiser('ordered')
        alpha=list(range(m.num_letters))
        var=obj.create_variables(k,alpha,m.num_states)
        cnf=obj.create_cnf(*var,m,k,alpha,True,False)
        with Solver(name='cadical195',bootstrap_with=cnf) as solver:
            sat=solver.solve()
            if sat:
                dfa=obj.construct_dfa_from_model(solver.get_model(),var[1],k,alpha)
                assert not sdfa_cegar_minimiser._verify_product(m,dfa,limit=1)[0]
        info.append((k,sat))
        if sat:break
    return {'lower':starting,'pre_ms':round(bound.elapsed_ms,3) if bound else 0,
            'search_seconds':round(perf_counter()-t0,4),'sizes':info}

for name in ('repo7','sparse2400','sparse4800'):
    m=build_3dfa(*make_sparse(name)) if name=='sparse4800' else prepare(name)[0]
    baseline=run(m,False)
    budgeted=run(m,True)
    assert baseline['sizes'][-1][1] and budgeted['sizes'][-1][1]
    assert baseline['sizes'][-1][0]==budgeted['sizes'][-1][0]
    print('COMPARE',json.dumps({'case':name,'states':m.num_states,
          'baseline':baseline,'cheap':budgeted}),flush=True)
