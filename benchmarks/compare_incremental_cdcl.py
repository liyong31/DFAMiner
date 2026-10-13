"""Native comparison of CEGAR, old CDCL theory, and incremental CDCL theory."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sdfa import sdfa
from cegar_minimiser import sdfa_cegar_minimiser
from cdcl_product import CDCLProductMinimiser

p=argparse.ArgumentParser()
p.add_argument('--repeats',type=int,default=5)
p.add_argument('--k',type=int,default=3)
p.add_argument('--filename',default='intermediate_3dfa.txt')
args=p.parse_args()
m=sdfa();m.load(args.filename)
for label in ['cegar','old-cdcl','incremental-cdcl']:
    for repeat in range(args.repeats):
        if label=='cegar': engine=sdfa_cegar_minimiser(verbose=False)
        else: engine=CDCLProductMinimiser(incremental=label=='incremental-cdcl',
                                          verbose=False,check_frequency=1)
        t=time.perf_counter()
        found,dfa=engine.solve('cadical195',args.k,list(range(m.num_letters)),m,True,False)
        elapsed=time.perf_counter()-t
        if found:
            bad,_=sdfa_cegar_minimiser._verify_product(m,dfa)
            assert not bad
        print('MEASURE',label,'k=',args.k,'repeat=',repeat,'sat=',found,
              'seconds=',round(elapsed,6),'details=',engine.stats[-1],flush=True)
