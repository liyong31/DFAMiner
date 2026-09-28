import itertools
import random
import unittest
from unittest.mock import patch

from conflict_graph import (_clique_bound, _not_k_colorable,
                            graph_lower_bound, incompatibility_graph)
from minimiser import sdfa_minimiser
from sdfa import sdfa
from strunion import word_type


def automaton(n, letters, edges, accepts=(), rejects=(), initials=(0,)):
    a = sdfa()
    a.set_num_states(n)
    a.set_num_letters(letters)
    a.init_states = set(initials)
    a.final_states = set(accepts)
    a.reject_states = set(rejects)
    for p, letter, q in edges:
        a.add_transition(p, letter, q)
    return a


class ConflictGraphTests(unittest.TestCase):
    def test_common_suffix_and_dontcare(self):
        a = automaton(5, 2, [(0, 0, 1), (0, 1, 2),
                              (1, 0, 3), (2, 0, 4)], (3,), (4,))
        states, edges = incompatibility_graph(a)
        self.assertTrue(edges[states.index(1)] & (1 << states.index(2)))
        self.assertEqual(graph_lower_bound(a), 2)
        # A missing transition is DONTCARE, not implicit rejection.
        a.trans[2].pop(0)
        _, edges = incompatibility_graph(a)
        self.assertFalse(edges[states.index(1)] & (1 << states.index(2)))
        self.assertEqual(graph_lower_bound(a), 1)

    def test_unreachable_conflicts_cannot_raise_bound(self):
        a = automaton(3, 0, [], (1,), (2,))
        self.assertEqual(incompatibility_graph(a), ([0], [0]))
        self.assertEqual(graph_lower_bound(a), 1)

    def test_exact_odd_cycle_and_budget_fallback(self):
        cycle = [((1 << ((i-1) % 5)) | (1 << ((i+1) % 5))) for i in range(5)]
        self.assertEqual(_clique_bound(cycle, 0), 2)
        self.assertTrue(_not_k_colorable(cycle, 2, 10000))
        self.assertFalse(_not_k_colorable(cycle, 2, 0))
        self.assertFalse(_not_k_colorable(cycle, 3, 10000))
        a = automaton(5, 0, [], initials=range(5))
        with patch("conflict_graph.incompatibility_graph", return_value=(list(range(5)), cycle)):
            self.assertEqual(graph_lower_bound(a), 3)
            self.assertEqual(graph_lower_bound(a, search_budget=0), 2)
            self.assertEqual(graph_lower_bound(a, exact_limit=4), 2)

    def test_skips_unsatisfiable_sizes_and_preserves_classification(self):
        # A complete unary mod-3 DFA has three pairwise incompatible states.
        a = automaton(3, 1, [(0, 0, 1), (1, 0, 2), (2, 0, 0)], (0,), (1, 2))
        self.assertEqual(graph_lower_bound(a), 3)
        minimizer = sdfa_minimiser()
        with patch.object(minimizer, 'solve', wraps=minimizer.solve) as solve:
            result = minimizer.minimise(a, 'minisat22', 1, 3, True, False)
        self.assertEqual([call.args[1] for call in solve.call_args_list], [3])
        for length in range(9):
            word = [0] * length
            self.assertEqual(result.run(0, word) == word_type.ACCEPT,
                             a.run(0, word) == word_type.ACCEPT)
        # The explicit user lower bound still takes priority.
        with patch.object(minimizer, 'solve', return_value=(True, result)) as solve:
            minimizer.minimise(a, 'minisat22', 4, 5, True, False)
        solve.assert_not_called()  # input-size cap is unchanged

    def test_small_random_instances_against_actual_sat_optimum(self):
        rng = random.Random(392)
        for _ in range(25):
            n = 3
            edges = [(p, c, rng.randrange(n)) for p in range(n)
                     for c in range(2) if rng.randrange(4)]
            labels = [rng.randrange(3) for _ in range(n)]
            a = automaton(n, 2, edges,
                          [p for p in range(n) if labels[p] == 1],
                          [p for p in range(n) if labels[p] == 2])
            lb = graph_lower_bound(a)
            optimum = next((k for k in range(1, n + 1)
                            if sdfa_minimiser().solve('minisat22', k, [0, 1], a,
                                                      True, False)[0]), None)
            self.assertIsNotNone(optimum)
            self.assertLessEqual(lb, optimum)
            result = sdfa_minimiser().minimise(a, 'minisat22', 1, n, True, False)
            self.assertEqual(result.num_states, optimum)
            for length in range(4):
                for word in itertools.product(range(2), repeat=length):
                    original = a.run(0, word)
                    if original != word_type.DONTCARE:
                        self.assertEqual(result.run(0, word) == word_type.ACCEPT,
                                         original == word_type.ACCEPT)


if __name__ == '__main__':
    unittest.main()
