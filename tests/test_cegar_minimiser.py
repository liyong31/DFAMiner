"""Regression checks: baseline versus opt-in CEGAR on the same 3DFA."""
import random
import unittest
from pathlib import Path

from cegar_minimiser import sdfa_cegar_minimiser
from minimiser import sdfa_minimiser
from sdfa import sdfa


def automaton(n, alphabet, arcs, pos=(), neg=(), roots=(0,)):
    a = sdfa()
    a.set_num_states(n)
    a.set_num_letters(alphabet)
    for q in roots:
        a.add_initial_state(q)
    for p, c, q in arcs:
        a.add_transition(p, c, q)
    for p in pos:
        a.add_final_state(p)
    for p in neg:
        a.add_reject_state(p)
    return a


class CEGARTests(unittest.TestCase):
    def compare(self, a, upper=3, bfs=False, safety=False):
        for k in range(1, min(upper, a.num_states) + 1):
            with self.subTest(k=k, bfs=bfs, safety=safety):
                baseline, _ = sdfa_minimiser().solve(
                    'minisat22', k, list(range(a.num_letters)), a, not bfs, safety)
                c = sdfa_cegar_minimiser(batch_size=8, seed_per_label=2, verbose=False)
                result, candidate = c.solve(
                    'minisat22', k, list(range(a.num_letters)), a, not bfs, safety)
                self.assertEqual(result, baseline)
                if result:
                    self.assertEqual(candidate.num_states, k)
                    self.assertFalse(c._verify_product(a, candidate)[0])

    def test_bundled_three_dfa(self):
        a = sdfa()
        a.load(str(Path(__file__).resolve().parents[1] / 'intermediate_3dfa.txt'))
        self.compare(a, bfs=False)
        self.compare(a, bfs=True)
        c = sdfa_cegar_minimiser(verbose=False)
        self.assertEqual(c.minimise(a, 'minisat22', 1, 8, True, False).num_states, 3)

    def test_shared_suffix(self):
        a = automaton(6, 2, [
            (0, 0, 1), (0, 1, 2), (1, 0, 3), (2, 1, 3),
            (1, 1, 4), (2, 0, 5)], pos=(3, 4), neg=(5,))
        self.compare(a)

    def test_multiple_roots_and_dontcare(self):
        a = automaton(4, 2, [(0, 0, 2), (1, 1, 3)],
                      pos=(2,), neg=(3,), roots=(0, 1))
        self.compare(a)

    def test_contradictory_state(self):
        a = automaton(1, 1, [], pos=(0,), neg=(0,))
        self.assertIsNone(sdfa_cegar_minimiser(verbose=False).minimise(
            a, 'minisat22', 1, 1, True, False))

    def test_random_instances(self):
        rng = random.Random(20261013)
        for trial in range(12):
            edges = [(p, a, rng.randrange(3)) for p in range(3)
                     for a in range(2) if rng.randrange(3)]
            labels = [rng.randrange(3) for _ in range(3)]
            a = automaton(3, 2, edges,
                          pos=[p for p, v in enumerate(labels) if v == 1],
                          neg=[p for p, v in enumerate(labels) if v == 2])
            with self.subTest(trial=trial):
                self.compare(a)

    def test_safety_constraints(self):
        a = automaton(3, 3, [
            (0, 0, 0), (0, 1, 1), (0, 2, 0),
            (1, 0, 0), (1, 1, 2), (1, 2, 0)],
            pos=(0, 1), neg=(2,))
        self.compare(a, safety=True)


if __name__ == '__main__':
    unittest.main()
