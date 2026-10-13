"""Regression coverage for lazy variable blocks and adaptive full-CNF fallback."""
import unittest
from cegar_minimiser import sdfa_cegar_minimiser
from sdfa import sdfa

class LazyCEGARTests(unittest.TestCase):
    def test_sparse_ids(self):
        a = sdfa()
        a.set_num_states(10000)
        a.set_num_letters(1)
        a.add_initial_state(0)
        a.add_reject_state(0)
        a.add_final_state(9999)
        a.add_transition(0, 0, 9999)
        c = sdfa_cegar_minimiser(seed_per_label=2, batch_size=4, verbose=False)
        sat, candidate = c.solve('minisat22', 2, [0], a, True, False)
        self.assertTrue(sat)
        self.assertLess(c.stats[-1]['variables'], 100)
        self.assertFalse(c._verify_product(a, candidate)[0])

    def test_adaptive_full_encoding(self):
        from pathlib import Path
        a = sdfa()
        a.load(str(Path(__file__).resolve().parents[1] / 'intermediate_3dfa.txt'))
        c = sdfa_cegar_minimiser(seed_per_label=1, batch_size=1,
                                 verbose=False, fallback_ratio=0.01)
        sat, candidate = c.solve('minisat22', 3, [0, 1], a, True, False)
        self.assertTrue(sat)
        self.assertTrue(c.stats[-1]['fallback_used'])
        self.assertFalse(c._verify_product(a, candidate)[0])

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            sdfa_cegar_minimiser(fallback_ratio=1.1)

if __name__ == '__main__':
    unittest.main()
