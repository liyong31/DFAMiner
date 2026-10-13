"""Regression tests for rollback-based CDCL reachability."""
import random
import unittest
from collections import deque
from incremental_product import IncrementalProductPropagator
from sdfa import sdfa

def make(seed):
    rng=random.Random(seed)
    a=sdfa()
    a.set_num_states(7)
    a.set_num_letters(2)
    a.add_initial_state(0)
    a.add_initial_state(3)
    for p in range(7):
        if rng.randrange(4)==0: a.add_final_state(p)
        if rng.randrange(4)==0: a.add_reject_state(p)
        for letter in range(2):
            if rng.randrange(4): a.add_transition(p,letter,rng.randrange(7))
    return a

def ids(k):
    result={}
    for p in range(k):
        for a in range(2):
            for q in range(k):
                result[p,a,q]=len(result)+1
    for p in range(k): result[p,-1,-1]=len(result)+1
    return result

def reference(a,selected):
    queue=deque((root,0) for root in a.init_states)
    reached=set(queue)
    while queue:
        p,q=queue.popleft()
        for letter,dst in a.trans[p].items():
            if (q,letter) not in selected: continue
            pair=(dst,selected[q,letter][0])
            if pair not in reached:
                reached.add(pair);queue.append(pair)
    return reached

class RollbackTests(unittest.TestCase):
    def test_random_nested_levels(self):
        for seed in range(120):
            a=make(seed)
            k=2+seed%3
            e=ids(k)
            prop=IncrementalProductPropagator(a,k,e,max_pairs=0)
            rng=random.Random(seed+5000)
            assignment=[(p,letter,rng.randrange(k)) for p in range(k) for letter in range(2)]
            rng.shuffle(assignment)
            def verify():
                self.assertEqual({divmod(pair,k) for pair in prop.parents},
                                 reference(a,prop.chosen))
            verify()
            prop.on_new_level()
            for p,letter,q in assignment[:len(assignment)//2]:
                prop.on_assignment(e[p,letter,q]);verify()
            prop.on_new_level()
            for p,letter,q in assignment[len(assignment)//2:]:
                prop.on_assignment(e[p,letter,q]);verify()
            prop.on_backtrack(1);verify()
            prop.on_backtrack(0);verify()

    def test_contradictory_empty_word(self):
        a=sdfa();a.set_num_states(1);a.set_num_letters(2)
        a.add_initial_state(0);a.add_final_state(0);a.add_reject_state(0)
        engine=IncrementalProductPropagator(a,1,ids(1))
        clause,propagated=engine._analyse_partial()
        self.assertEqual(clause,[])
        self.assertEqual(propagated,[])

    def test_native_bundle(self):
        from pathlib import Path
        from cdcl_product import CDCLProductMinimiser
        from cegar_minimiser import sdfa_cegar_minimiser
        a=sdfa()
        a.load(str(Path(__file__).resolve().parents[1]/'intermediate_3dfa.txt'))
        obj=CDCLProductMinimiser(incremental=True,check_frequency=1,verbose=False)
        found=obj.minimise(a,'cadical195',1,8,True,False)
        self.assertIsNotNone(found)
        self.assertEqual(found.num_states,3)
        self.assertFalse(sdfa_cegar_minimiser._verify_product(a,found,limit=1)[0])

if __name__=='__main__':
    unittest.main()
