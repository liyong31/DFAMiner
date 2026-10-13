"""Cheap, sound lower bounds via bounded distinguishing-suffix signatures.

No quadratic residual-pair matrix is constructed. A suffix is retained only
with its exact ACCEPT/REJECT observations. Opposite observations certify an
incompatible pair, and any clique of such pairs lower-bounds DFA size.

The work is bounded by a finite number of suffixes, candidate states, and
word transitions; the optional wall-clock budget ends the search early.
"""
from collections import deque
from time import perf_counter


class CheapIncompatibilityBound:
    def __init__(self, machine, max_suffixes=32, max_candidates=256,
                 max_depth=8, budget_ms=25):
        if min(max_suffixes, max_candidates, max_depth) < 1 or budget_ms < 0:
            raise ValueError('invalid lightweight lower-bound budget')
        self.machine = machine
        self.max_suffixes = max_suffixes
        self.max_candidates = max_candidates
        self.max_depth = max_depth
        self.budget_ms = budget_ms
        self.suffixes = []
        self.clique = []
        self.witnesses = {}
        self.elapsed_ms = 0
        self._run()

    def _run(self):
        start = perf_counter()
        deadline = start + self.budget_ms / 1000 if self.budget_ms else float('inf')
        m = self.machine
        reachable = sorted(m.get_reachable_states())
        if not reachable:
            self.elapsed_ms = (perf_counter()-start)*1000
            return
        # Candidate pool includes both labelled states and evenly distributed
        # reachable residuals; reserve coverage of positive and negative states.
        labelled = sorted((m.final_states | m.reject_states) & set(reachable))
        candidate = []
        seen = set()
        def add(p):
            if p in seen or len(candidate) >= self.max_candidates:
                return
            candidate.append(p)
            seen.add(p)
        for p in labelled[:max(1, self.max_candidates//2)]:
            add(p)
        stride = max(1, len(reachable)//max(1,self.max_candidates))
        for p in reachable[::stride]:
            add(p)
        for p in reachable:
            if len(candidate)>=self.max_candidates: break
            add(p)
        # Signatures are integers marking suffix acceptance/rejection.
        positive = [0]*len(candidate)
        negative = [0]*len(candidate)
        suffix_queue = deque([()])
        visited = {()}
        while suffix_queue and len(self.suffixes)<self.max_suffixes:
            if perf_counter() >= deadline: break
            suffix = suffix_queue.popleft()
            pos = neg = 0
            for i,p in enumerate(candidate):
                state = p
                for a in suffix:
                    state = m.trans[state].get(a)
                    if state is None: break
                if state is None: continue
                if state in m.final_states: pos |= 1<<i
                if state in m.reject_states: neg |= 1<<i
            if pos and neg:
                bit = len(self.suffixes)
                self.suffixes.append(suffix)
                for i in self._bits(pos): positive[i] |= 1 << bit
                for i in self._bits(neg): negative[i] |= 1 << bit
            if len(suffix) < self.max_depth:
                for a in range(m.num_letters):
                    w = (a,)+suffix
                    if w not in visited:
                        visited.add(w)
                        suffix_queue.append(w)
        incompatible = [0]*len(candidate)
        for i in range(len(candidate)):
            for j in range(i):
                if (positive[i]&negative[j]) or (negative[i]&positive[j]):
                    incompatible[i] |= 1<<j
                    incompatible[j] |= 1<<i
        degree = [v.bit_count() for v in incompatible]
        best=[]
        for seed in sorted(range(len(candidate)),key=lambda i:(-degree[i],i))[:min(len(candidate),24)]:
            clique=[seed]
            avail=incompatible[seed]
            while avail:
                j=max(self._bits(avail),key=lambda q:(degree[q],-q))
                clique.append(j)
                avail &= incompatible[j]
            if len(clique)>len(best):best=clique
        self.clique = [candidate[i] for i in best]
        for i,p in enumerate(self.clique):
            for q in self.clique[i+1:]:
                a=candidate.index(p);b=candidate.index(q)
                common=(positive[a]&negative[b]) | (negative[a]&positive[b])
                if not common:raise AssertionError('clique without distinguishing suffix')
                self.witnesses[p,q] = self.suffixes[(common&-common).bit_length()-1]
        self.elapsed_ms=(perf_counter()-start)*1000

    @staticmethod
    def _bits(bits):
        while bits:
            low=bits&-bits
            yield low.bit_length()-1
            bits ^= low

    @property
    def lower_bound(self):
        return max(1,len(self.clique))

    def verify(self):
        m=self.machine
        reachable=m.get_reachable_states()
        assert all(q in reachable for q in self.clique)
        for (p,q), suffix in self.witnesses.items():
            def label(s):
                for a in suffix:
                    s=m.trans[s].get(a)
                    if s is None:return 0
                return 1 if s in m.final_states else (-1 if s in m.reject_states else 0)
            if label(p)*label(q)!=-1:
                raise AssertionError((p,q,suffix))
        assert len(self.witnesses)==len(self.clique)*(len(self.clique)-1)//2
        return True
