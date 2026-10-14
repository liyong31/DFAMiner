"""Differential soundness of source trimming and RGS state-label symmetry."""
import random
import unittest
from sdfa import sdfa
from preprocess_3dfa import (trim_observation_irrelevant,
                              rgs_symmetry_clauses)
from preprocessed_eager import PreprocessedEagerMinimiser
from pysat.solvers import Solver
from cegar_minimiser import sdfa_cegar_minimiser


def solve(m,k,variant):
    eng=PreprocessedEagerMinimiser('both',trim=False,
        symmetry='bfs' if variant=='rgs' else 'none')
    vars=eng.create_variables(k,list(range(m.num_letters)),m.num_states)
    cnf=eng.create_cnf(*vars,m,k,list(range(m.num_letters)),True,False)
    with Solver(name='cadical195',bootstrap_with=cnf) as solver:
        sat=solver.solve()
        if sat:
            candidate=eng.construct_dfa_from_model(
                solver.get_model(),vars[1],k,list(range(m.num_letters)))
            assert not sdfa_cegar_minimiser._verify_product(m,candidate,limit=1)[0]
        return sat


class Soundness(unittest.TestCase):
    def test_random_cyclic(self):
        for seed in range(30):
            r=random.Random(9000+seed)
            m=sdfa()
            n=r.randrange(2,8)
            m.set_num_states(n)
            m.set_num_letters(2)
            m.add_initial_state(0)
            for q in range(n):
                label=r.randrange(3)
                if label==1:m.add_final_state(q)
                if label==2:m.add_reject_state(q)
                for a in range(2):
                    if r.randrange(4):
                        m.add_transition(q,a,r.randrange(n))
            trimmed,_=trim_observation_irrelevant(m)
            for k in (1,2,3):
                baseline=solve(m,k,'base')
                self.assertEqual(solve(m,k,'rgs'),baseline,(seed,k))
                self.assertEqual(solve(trimmed,k,'base'),baseline,(seed,k))
                self.assertEqual(solve(trimmed,k,'rgs'),baseline,(seed,k))

    def test_multiple_initial(self):
        m=sdfa()
        m.set_num_states(5);m.set_num_letters(2)
        m.add_initial_state(0);m.add_initial_state(1)
        m.add_transition(0,0,2);m.add_transition(1,0,2)
        m.add_transition(2,1,3);m.add_transition(3,0,2)
        m.add_final_state(2);m.add_reject_state(3)
        m.add_transition(4,0,4)
        trimmed,meta=trim_observation_irrelevant(m)
        self.assertEqual(meta['removed_states'],1)
        for k in (1,2,3):
            b=solve(m,k,'base')
            self.assertEqual(solve(m,k,'rgs'),b)
            self.assertEqual(solve(trimmed,k,'base'),b)
            self.assertEqual(solve(trimmed,k,'rgs'),b)
