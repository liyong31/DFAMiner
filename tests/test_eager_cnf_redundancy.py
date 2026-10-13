"""Regression tests for eager SAT redundancy removal.

All variants must agree on SAT/UNSAT and classify the full input automaton.
The positive/negative cross-pair clauses are redundant by propositional
resolution; ordered transition-pair clauses are literal duplicates.
"""
import random
import unittest
from pathlib import Path

from sdfa import sdfa
from cegar_minimiser import sdfa_cegar_minimiser
from minimiser import sdfa_minimiser
from eager_cnf_variants import ReducedEagerMinimiser


def random_machine(seed):
    rng = random.Random(seed)
    n = rng.randrange(2, 6)
    alphabet = 2
    a = sdfa()
    a.set_num_states(n)
    a.set_num_letters(alphabet)
    a.add_initial_state(0)
    for q in range(n):
        label = rng.randrange(3)
        if label == 1:
            a.add_final_state(q)
        elif label == 2:
            a.add_reject_state(q)
        for symbol in range(alphabet):
            if rng.randrange(4):
                a.add_transition(q, symbol, rng.randrange(n))
    return a


class EagerRedundancyTests(unittest.TestCase):
    def _check(self, machine, k, bfs, safety):
        alphabet = list(range(machine.num_letters))
        original = sdfa_minimiser()
        variables = original.create_variables(k, alphabet, machine.num_states)
        base = original.create_cnf(*variables, machine, k, alphabet, not bfs, safety)
        with self.subTest(k=k,bfs=bfs,safety=safety):
            from pysat.solvers import Solver
            with Solver(name='cadical195', bootstrap_with=base) as solver:
                baseline_sat = solver.solve()
            for mode in ('ordered','implied','both','dedup'):
                modified = ReducedEagerMinimiser(mode)
                v = modified.create_variables(k,alphabet,machine.num_states)
                clauses = modified.create_cnf(*v, machine, k, alphabet, not bfs, safety)
                self.assertLessEqual(len(clauses),len(base))
                with Solver(name='cadical195',bootstrap_with=clauses) as solver:
                    sat = solver.solve()
                    self.assertEqual(sat,baseline_sat,mode)
                    if sat:
                        dfa = modified.construct_dfa_from_model(solver.get_model(),v[1],k,alphabet)
                        self.assertFalse(sdfa_cegar_minimiser._verify_product(machine,dfa,limit=1)[0])

    def test_bundled(self):
        machine=sdfa()
        machine.load(str(Path(__file__).resolve().parents[1]/'intermediate_3dfa.txt'))
        for bfs in (False,True):
            for k in (1,2,3):
                self._check(machine,k,bfs,False)

    def test_random(self):
        for seed in range(15):
            machine=random_machine(seed)
            for k in (1,2,3):
                self._check(machine,k,False,False)

    def test_safety(self):
        machine=sdfa()
        machine.set_num_states(3)
        machine.set_num_letters(3)
        machine.add_initial_state(0)
        for p,a,q in ((0,0,0),(0,1,1),(0,2,0),(1,0,0),(1,1,2),(1,2,0)):
            machine.add_transition(p,a,q)
        machine.add_final_state(0)
        machine.add_final_state(1)
        machine.add_reject_state(2)
        for k in (3,4):
            self._check(machine,k,False,True)


if __name__ == '__main__':
    unittest.main()
