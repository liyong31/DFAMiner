"""Compare incremental 3DFA incompatibility with independent product BFS."""
import random
import unittest
from collections import deque
from strunion import dfa_builder, word_type
from incompatibility import IncompatibilityInfo
from cegar_minimiser import sdfa_cegar_minimiser
from incompatibility_eager import IncompatibilityEagerMinimiser


def construct(samples, collected):
    b=dfa_builder(collect_incompatibility=collected)
    for word,label in sorted(samples):
        b.add(word,word_type.ACCEPT if label==1 else word_type.REJECT)
    return dfa_builder.build(b,2)


def independently_incompatible(m,p,q):
    todo=deque([(p,q)]);seen={(p,q)}
    while todo:
        a,b=todo.popleft()
        if ((a in m.final_states and b in m.reject_states) or
            (a in m.reject_states and b in m.final_states)):
            return True
        for letter,dst in m.trans[a].items():
            other=m.trans[b].get(letter)
            if other is not None and (dst,other) not in seen:
                seen.add((dst,other));todo.append((dst,other))
    return False


class CollectedPairTests(unittest.TestCase):
    def check(self,samples):
        yes=construct(samples,True)
        no=construct(samples,False)
        self.assertEqual(yes.num_states,no.num_states)
        self.assertEqual(yes.trans,no.trans)
        self.assertEqual(yes.final_states,no.final_states)
        self.assertEqual(yes.reject_states,no.reject_states)
        info=yes.incompatibility
        self.assertTrue(info.verify())
        for p in range(yes.num_states):
            for q in range(yes.num_states):
                self.assertEqual(bool(info.adjacency[p] & (1<<q)),
                                 independently_incompatible(yes,p,q),
                                 (p,q,samples))
        for k in range(1,5):
            baseline=sdfa_cegar_minimiser(verbose=False)
            a,_=baseline.solve('cadical195',k,[0,1],yes,True,False)
            if len(info.clique)>k:
                self.assertFalse(a)

    def test_manual(self):
        self.check([((0,0),1),((0,1),-1),((1,0),-1),((1,1),1)])
        self.check([((0,0,1),1),((1,0,1),-1),((1,1,0),1)])
        self.check([((0,),1),((1,),-1)])

    def test_random_samples(self):
        for seed in range(35):
            r=random.Random(seed)
            samples={}
            for _ in range(6+seed%12):
                word=tuple(r.randrange(2) for j in range(r.randrange(1,9)))
                samples[word]=1 if r.randrange(2) else -1
            self.check(list(samples.items()))

    def test_empty_word_root_label(self):
        # Root classification is assigned before canonicalisation.
        for sign in (-1, 1):
            builder=dfa_builder(collect_incompatibility=True)
            builder.add((0,),word_type.ACCEPT)
            builder.add((1,),word_type.REJECT)
            builder.root.is_final=(word_type.ACCEPT if sign==1 else word_type.REJECT)
            machine=dfa_builder.build(builder,2)
            self.assertTrue(machine.incompatibility.verify())
            for p in range(machine.num_states):
                for q in range(machine.num_states):
                    self.assertEqual(bool(machine.incompatibility.adjacency[p] & (1<<q)),
                                     independently_incompatible(machine,p,q))

    def test_solver_selected_pairs(self):
        for seed in range(9):
            rng=random.Random(900+seed)
            samples={}
            for _ in range(15):
                word=tuple(rng.randrange(2) for _ in range(rng.randrange(1,8)))
                samples[word]=1 if rng.randrange(2) else -1
            m=construct(samples.items(),True)
            for k in range(1,5):
                base=sdfa_cegar_minimiser(verbose=False)
                sat,_=base.solve('cadical195',k,[0,1],m,True,False)
                for budget in (0,10,100):
                    opt=IncompatibilityEagerMinimiser(pair_budget=budget,verbose=False)
                    result,_=opt.solve('cadical195',k,[0,1],m,True,False)
                    self.assertEqual(sat,result)
