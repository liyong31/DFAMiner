"""Differential checks for compact product verification."""
from collections import deque
import random
import unittest
from cegar_minimiser import sdfa_cegar_minimiser
from sdfa import sdfa

def reference_check(m, d):
    queue = deque((root, 0) for root in m.init_states)
    reached = set(queue)
    while queue:
        p, q = queue.popleft()
        if ((p in m.final_states and q not in d.final_states) or
                (p in m.reject_states and q in d.final_states)):
            return True
        for letter, dst in m.trans[p].items():
            pair = (dst, d.trans[q][letter])
            if pair not in reached:
                reached.add(pair)
                queue.append(pair)
    return False

class CompactVerifierTests(unittest.TestCase):
    def test_random(self):
        rng = random.Random(13579)
        for trial in range(300):
            n, alphabet, k = rng.randrange(1,20), rng.randrange(1,5), rng.randrange(1,9)
            m, d = sdfa(), sdfa()
            m.set_num_states(n); m.set_num_letters(alphabet)
            d.set_num_states(k); d.set_num_letters(alphabet)
            d.add_initial_state(0)
            for q in range(k):
                if rng.randrange(2): d.add_final_state(q)
                for a in range(alphabet): d.add_transition(q,a,rng.randrange(k))
            for p in range(n):
                label = rng.randrange(5)
                if label in (1,3): m.add_final_state(p)
                if label in (2,3): m.add_reject_state(p)
                for a in range(alphabet):
                    if rng.randrange(4):
                        m.add_transition(p,a,rng.randrange(n) if trial%2 else rng.randrange(p,n))
            for root in rng.sample(range(n),rng.randrange(1,min(n,3)+1)):
                m.add_initial_state(root)
            examples, count = sdfa_cegar_minimiser._verify_product(m,d,limit=7)
            with self.subTest(trial=trial):
                self.assertEqual(bool(examples),reference_check(m,d))
                self.assertGreaterEqual(count,1)
                for root, word in examples:
                    self.assertIn(root,m.init_states)
                    p,q=root,0
                    for a in word:
                        self.assertIn(a,m.trans[p])
                        p,q=m.trans[p][a],d.trans[q][a]
                    self.assertTrue((p in m.final_states and q not in d.final_states) or
                                    (p in m.reject_states and q in d.final_states))

    def test_zero_alphabet(self):
        m,d=sdfa(),sdfa()
        m.set_num_states(1);m.set_num_letters(0);m.add_initial_state(0);m.add_final_state(0)
        d.set_num_states(1);d.set_num_letters(0);d.add_initial_state(0)
        self.assertEqual(sdfa_cegar_minimiser._verify_product(m,d), ([(0,())],1))
        with self.assertRaises(ValueError):
            sdfa_cegar_minimiser._verify_product(m,d,limit=0)

if __name__ == '__main__':
    unittest.main()
