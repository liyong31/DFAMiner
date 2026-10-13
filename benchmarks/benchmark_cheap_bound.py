"""Native benchmark of certified cheap lower bounds vs full pair collection."""
import sys
from pathlib import Path
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.compare_large_3dfa import prepare
from cheap_lower_bound import CheapIncompatibilityBound

for name in ('repo7','repo8','repo10','sparse2400','sparse4800'):
    if name=='sparse4800':
        from benchmarks.compare_large_3dfa import make_sparse,build_3dfa
        m=build_3dfa(*make_sparse(name))
    else:
        m,_=prepare(name)
    print('CASE',name,'N',m.num_states,flush=True)
    for ms in (10,25,75):
        b=CheapIncompatibilityBound(m,budget_ms=ms,max_suffixes=64,
                                    max_candidates=256,max_depth=9)
        b.verify()
        print('BOUND',name,ms,b.lower_bound,len(b.suffixes),
              round(b.elapsed_ms,3),flush=True)
