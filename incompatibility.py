"""Exact pairwise residual incompatibility collected at 3DFA registration.

For finite-word three-valued acyclic DFAs, two residual states are
incompatible iff their labels conflict or some common-letter successors are
incompatible. The incremental lexical builder registers a state only after
its children have become canonical. This permits one exact bottom-up test
per pair, and avoids product search after the 3DFA is constructed.

All bitsets use canonical construction IDs. Export remaps into the final
DFA's state numbering and restricts clique certificates to root-reachable
states. Neither bitset metadata nor the clique is used to alter the language.
"""
from collections import deque


class PairwiseCollector:
    def __init__(self):
        self.states = []
        self.identifiers = {}
        self.rows = []                 # rows[i]: incompatible j < i
        self.pairs_checked = 0

    def register(self, state):
        sid = id(state)
        if sid in self.identifiers:
            return self.identifiers[sid]
        children = {}
        for letter, child in zip(state.labels, state.states):
            if id(child) not in self.identifiers:
                raise AssertionError('Register children before parents')
            children[letter] = self.identifiers[id(child)]
        label = state.is_final.value
        idx = len(self.states)
        row = 0
        for j, (other_label, other) in enumerate(self.states):
            self.pairs_checked += 1
            incompatible = (label != 0 and other_label != 0 and label != other_label)
            if not incompatible:
                for a, child in children.items():
                    sibling = other.get(a)
                    if sibling is None or child == sibling:
                        continue
                    hi, lo = (child, sibling) if child > sibling else (sibling, child)
                    if (self.rows[hi] >> lo) & 1:
                        incompatible = True
                        break
            if incompatible:
                row |= 1 << j
        self.identifiers[sid] = idx
        self.states.append((label, children))
        self.rows.append(row)
        return idx

    def finalise(self, builder, root):
        # Root is generally not in the canonical register. At this point all
        # its children have been registered; finalise it exactly once.
        self.register(root)

    def export(self, machine, builder):
        # The 3DFA converter traverses canonical state objects and assigns
        # output IDs depth-first. Reproduce its ordering, without modifying it.
        order = {}
        def visit(state):
            if id(state) in order:
                return
            order[id(state)] = len(order)
            for child in state.states:
                visit(child)
        visit(builder.root)
        if len(order) != machine.num_states:
            raise AssertionError((len(order), machine.num_states))
        mapped = [None] * machine.num_states
        for state_id, target_id in order.items():
            if state_id not in self.identifiers:
                raise AssertionError('Missing canonical state in incompatibility collector')
            mapped[target_id] = self.identifiers[state_id]
        reverse = {old: new for new, old in enumerate(mapped)}
        adjacency = [0] * machine.num_states
        for new, old in enumerate(mapped):
            row = self.rows[old]
            while row:
                lowest = row & -row
                partner = lowest.bit_length() - 1
                if partner in reverse:
                    peer = reverse[partner]
                    adjacency[new] |= 1 << peer
                    adjacency[peer] |= 1 << new
                row ^= lowest
        return IncompatibilityInfo(machine, adjacency, self.pairs_checked)


class IncompatibilityInfo:
    def __init__(self, machine, adjacency, pairs_checked=0):
        self.machine = machine
        self.adjacency = adjacency
        self.pairs_checked = pairs_checked
        self.reachable = set(machine.get_reachable_states())
        self.clique = self._greedy_clique()

    def _greedy_clique(self):
        mask = 0
        for i in self.reachable:
            mask |= 1 << i
        if not mask:
            return []
        # Multiple greedy seeds improve the certificate without an
        # exponential maximum-clique algorithm.
        degrees = [(self.adjacency[i] & mask).bit_count() for i in range(len(self.adjacency))]
        candidates = sorted(self.reachable, key=lambda i: (-degrees[i], i))
        best = []
        for seed in candidates[:min(24, len(candidates))]:
            result = [seed]
            rest = self.adjacency[seed] & mask
            while rest:
                next_node = max(self._bits(rest), key=lambda v: (degrees[v], -v))
                result.append(next_node)
                rest &= self.adjacency[next_node]
            if len(result) > len(best):
                best = result
        return best

    @staticmethod
    def _bits(mask):
        while mask:
            low = mask & -mask
            yield low.bit_length() - 1
            mask ^= low

    def selected_pairs(self, budget=0):
        """Return sound pairs, prioritising certified clique edges.

        Budget 0 means all clique pairs only. Positive budgets also include
        highest-degree remaining pairs, bounded independently of input size.
        """
        clique = self.clique
        pairs = {(min(a,b), max(a,b))
                 for i,a in enumerate(clique) for b in clique[i+1:]}
        if budget <= 0:
            return sorted(pairs)
        reachable_mask = sum(1 << i for i in self.reachable)
        ordered = sorted(self.reachable,
                         key=lambda i: -((self.adjacency[i] & reachable_mask).bit_count()))
        for i in ordered:
            for j in self._bits(self.adjacency[i] & reachable_mask):
                if i < j:
                    pairs.add((i, j))
                    if len(pairs) >= budget + len(clique)*(len(clique)-1)//2:
                        return sorted(pairs)
        return sorted(pairs)

    def verify(self):
        for p, neighbors in enumerate(self.adjacency):
            if neighbors & (1 << p):
                raise AssertionError('Self-incompatibility')
            for q in self._bits(neighbors):
                if not (self.adjacency[q] & (1 << p)):
                    raise AssertionError('Asymmetric incompatibility')
        for i,p in enumerate(self.clique):
            if p not in self.reachable:
                raise AssertionError('Unreachable clique member')
            for q in self.clique[i+1:]:
                if not ((self.adjacency[p] >> q) & 1):
                    raise AssertionError('Invalid clique certificate')
        return True
