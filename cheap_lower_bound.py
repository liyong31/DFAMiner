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
        # Bounded witness signatures are reused for SAT lemmas.
        self.candidates = ()
        self.positive_signatures = ()
        self.negative_signatures = ()
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
        # Keep only O(max_candidates * max_suffixes) witness information;
        # unlike the collector used by the full quadratic implementation,
        # this does not retain input-state pair data.
        self.candidates = tuple(candidate)
        self.positive_signatures = tuple(positive)
        self.negative_signatures = tuple(negative)
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


    def scored_pairs(self, limit=300, budget_ms=15, policy='impact'):
        """Choose sound nontrivial incompatibility lemmas without full O(N^2).

        The bounded signature pool has at most max_candidates states. A pair is
        retained only if a concrete common suffix gives opposite mandatory
        labels. Opposite directly-labelled pairs are skipped: their clauses
        are already generated in the original eager encoding.

        Policy 'impact' scores shared predecessor counts, witnessed degree
        and short suffixes; 'short' favours shorter witnesses; 'degree'
        emphasises the number of observed incompatible neighbours.
        Rankings are heuristics: *all* selected pairs are independently sound.

        The time limit is soft (checked during pair enumeration), and always
        affects only the number of returned clauses, never soundness.
        """
        from time import perf_counter
        if limit < 0 or budget_ms < 0:
            raise ValueError('negative scored-pair budget')
        if policy not in ('impact', 'short', 'degree', 'fanin'):
            raise ValueError('unknown scored-pair policy: ' + str(policy))
        if limit == 0 or not self.candidates or budget_ms == 0:
            return []
        started = perf_counter()
        deadline = started + budget_ms / 1000.0
        machine = self.machine
        cand = self.candidates
        psig = self.positive_signatures
        nsig = self.negative_signatures
        suffixes = self.suffixes
        fanin = [0] * machine.num_states
        for transitions in machine.trans:
            for dst in transitions.values():
                fanin[dst] += 1
        # Only the bounded candidate-pool graph is considered; no state-pair
        # computation proportional to |Q|^2 is performed.
        discovered = []
        degree = [0] * len(cand)
        for i,p in enumerate(cand):
            if (i & 31) == 0 and perf_counter() >= deadline:
                break
            for j in range(i):
                q = cand[j]
                # Already explicitly present as label-incompatibility clauses.
                if ((p in machine.final_states and q in machine.reject_states)
                    or (p in machine.reject_states and q in machine.final_states)):
                    continue
                common = (psig[i] & nsig[j]) | (nsig[i] & psig[j])
                if not common:
                    continue
                # Select a shortest available distinguishing suffix. The empty
                # suffix cannot occur after the directly-labelled pair check.
                witness_index = min(self._bits(common),
                                    key=lambda index: (len(suffixes[index]), index))
                suffix = suffixes[witness_index]
                if not suffix:
                    continue
                discovered.append((i,j,witness_index,fanin[p],fanin[q]))
                degree[i] += 1
                degree[j] += 1
        if not discovered:
            return []

        def ranking(item):
            i,j,s,fi,fj = item
            length = len(suffixes[s])
            di,dj = degree[i],degree[j]
            if policy == 'short':
                return (-length, di+dj, fi+fj)
            if policy == 'degree':
                return (di+dj, -length, fi+fj)
            if policy == 'fanin':
                return ((1+fi)*(1+fj), -length, di+dj)
            # Stronger fan-in and degree may make the implication useful
            # under more SAT decisions. This is *not* a correctness property.
            return (((1+min(fi,8))*(1+min(fj,8))*
                     (1+di)*(1+dj)) // (1+length),
                    -length, di+dj)
        discovered.sort(key=lambda item:(ranking(item),
                                          -min(cand[item[0]],cand[item[1]]),
                                          -max(cand[item[0]],cand[item[1]])),
                        reverse=True)

        # Avoid spending the entire lemma budget on a few heavily shared
        # states. If the graph is sparse, fill the remainder without caps.
        cap = max(8, (4*limit + len(cand)-1)//max(1,len(cand)))
        count = [0]*len(cand)
        selected = []
        used = set()
        for constrained in (True, False):
            for item in discovered:
                if len(selected) >= limit:
                    break
                i,j,s,_,_ = item
                if (i,j) in used:
                    continue
                if constrained and (count[i] >= cap or count[j] >= cap):
                    continue
                p,q = cand[i],cand[j]
                selected.append((min(p,q),max(p,q),suffixes[s]))
                used.add((i,j))
                count[i] += 1
                count[j] += 1
            if len(selected) >= limit:
                break
        return selected

    def verify_scored_pairs(self, pairs):
        """Verify each selected witness independently from the source 3DFA."""
        m = self.machine
        seen = set()
        reachable = m.get_reachable_states()
        for p,q,w in pairs:
            if not (p < q and p in reachable and q in reachable):
                raise AssertionError(('bad reachable pair',p,q))
            if (p,q) in seen:
                raise AssertionError(('duplicate pair',p,q))
            seen.add((p,q))
            def value(state):
                for letter in w:
                    state = m.trans[state].get(letter)
                    if state is None:
                        return 0
                if state in m.final_states:
                    return 1
                if state in m.reject_states:
                    return -1
                return 0
            if value(p)*value(q) != -1:
                raise AssertionError(('incorrect pair witness',p,q,w))
        return True
