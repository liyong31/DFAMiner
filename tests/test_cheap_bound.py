"""Differential verification of cheap lower bound against exact eager SAT."""
import random
import unittest
from pathlib import Path
from sdfa import sdfa
from cheap_lower_bound import CheapIncompatibilityBound
from cheap_bound_eager import CheapBoundEagerMinimiser
from cegar_minimiser import sdfa_cegar_minimiser


class CheapBoundTests(unittest.TestCase):
    def test_random_sound(self):
        for seed in range(45):
            rng = random.Random(seed)
            m = sdfa()
            n = rng.randrange(2, 9)
            m.set_num_states(n)
            m.set_num_letters(2)
            m.add_initial_state(0)
            for p in range(n):
                label = rng.randrange(3)
                if label == 1: m.add_final_state(p)
                if label == 2: m.add_reject_state(p)
                for a in range(2):
                    if rng.randrange(4): m.add_transition(p, a, rng.randrange(n))
            info = CheapIncompatibilityBound(m,budget_ms=500,max_depth=8)
            self.assertTrue(info.verify())
            for k in range(1, min(n, info.lower_bound)):
                sat,_ = sdfa_cegar_minimiser(verbose=False).solve('cadical195',k,[0,1],m,True,False)
                self.assertFalse(sat,(seed,k,info.lower_bound))

    def test_bundle_optimum(self):
        m=sdfa()
        m.load(str(Path(__file__).resolve().parents[1]/'intermediate_3dfa.txt'))
        bound=CheapIncompatibilityBound(m,budget_ms=100)
        self.assertTrue(bound.verify())
        self.assertLessEqual(bound.lower_bound,3)
        result=CheapBoundEagerMinimiser(verbose=False).minimise(
            m,'cadical195',1,8,True,False)
        self.assertEqual(result.num_states,3)
