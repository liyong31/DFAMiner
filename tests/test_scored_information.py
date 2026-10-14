"""Verify sound scored-pair witnesses and eager SAT equivalence."""
import random
import unittest
from pathlib import Path
from sdfa import sdfa
from cheap_lower_bound import CheapIncompatibilityBound
from cheap_bound_eager import CheapBoundEagerMinimiser
from eager_cnf_variants import ReducedEagerMinimiser
from pysat.solvers import Solver
from cegar_minimiser import sdfa_cegar_minimiser

class ScoredInformationTests(unittest.TestCase):
    def test_pairs_random(self):
        for seed in range(35):
            rng=random.Random(seed)
            m=sdfa()
            n=rng.randrange(3,9)
            m.set_num_states(n); m.set_num_letters(2)
            m.add_initial_state(0)
            for p in range(n):
                typ=rng.randrange(3)
                if typ==1: m.add_final_state(p)
                if typ==2: m.add_reject_state(p)
                for a in range(2):
                    if rng.randrange(5):m.add_transition(p,a,rng.randrange(n))
            info=CheapIncompatibilityBound(m, budget_ms=100, max_candidates=256)
            self.assertTrue(info.verify())
            for policy in ('impact','short','degree','fanin'):
                pairs=info.scored_pairs(limit=30,budget_ms=100,policy=policy)
                self.assertTrue(info.verify_scored_pairs(pairs))

    def test_sat_equivalence(self):
        m=sdfa()
        m.load(str(Path(__file__).resolve().parents[1]/'intermediate_3dfa.txt'))
        for k in (1,2,3):
            alpha=list(range(m.num_letters))
            outcomes=[]
            for pairs in (0,40):
                worker=CheapBoundEagerMinimiser(pair_budget=pairs,verbose=False,
                                                budget_ms=100,pair_budget_ms=100)
                cert=CheapIncompatibilityBound(m,budget_ms=100)
                worker.selected_pairs=cert.scored_pairs(limit=pairs,budget_ms=100)
                cert.verify_scored_pairs(worker.selected_pairs)
                v=worker.create_variables(k,alpha,m.num_states)
                cnf=worker.create_cnf(*v,m,k,alpha,True,False)
                with Solver(name='cadical195',bootstrap_with=cnf) as sol:
                    sat=sol.solve()
                    outcomes.append(sat)
                    if sat:
                        target=worker.construct_dfa_from_model(sol.get_model(),v[1],k,alpha)
                        self.assertFalse(sdfa_cegar_minimiser._verify_product(m,target,limit=1)[0])
            self.assertEqual(outcomes[0],outcomes[1])
