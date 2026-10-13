"""Backtrackable incremental product reachability for CDCL DFA inference.

Only positive target-edge assignments extend the product. Each reached pair has
one causal predecessor, so every conflict and propagation has an explanation
in terms of actually assigned DFA transitions. Shared-suffix 3DFA states may
reach several target DFA states; there is no one-hot product-state mapping.

The superclass retains the exact complete-model verifier. Partial traversal
can be capped without compromising soundness or completeness of the solver.
"""
from collections import defaultdict, deque

from cdcl_product import DFAProductPropagator


class IncrementalProductPropagator(DFAProductPropagator):
    """Maintain a trail of reached input/target pairs across CDCL decisions."""

    def __init__(self, input_sdfa, n, edges, max_pairs=12000,
                 check_frequency=4, lookahead_budget=32,
                 suffix_budget=1000):
        super().__init__(input_sdfa, n, edges, max_pairs,
                         check_frequency, lookahead_budget, suffix_budget)
        self.chosen = {}                # (target state, letter) -> (dst, selector)
        self.parents = {}               # product ID -> (parent ID, selector) or None
        self.pair_trail = []
        self.pair_marks = []            # pair-trail length at each decision level
        self._late_fixed = False
        self.watch = defaultdict(list)  # (target state, letter) -> product IDs
        self.by_target = [[] for _ in range(n)]
        self.positive = [[] for _ in range(n)]
        self.negative = [[] for _ in range(n)]
        self.stats.update(
            pair_additions=0, pair_rollbacks=0, peak_pairs=0,
            edge_activations=0, watcher_visits=0,
            saturated_pairs=0, max_pair_skips=0, reach_rebuilds=0)
        self._initialise_roots()

    def _initialise_roots(self):
        todo = deque()
        for root in sorted(self.input.init_states):
            self._add_pair(root, 0, None, None, todo, is_root=True)
        self._drain(todo)

    def _add_pair(self, p, q, predecessor, selector, todo, is_root=False):
        pair = p * self.n + q
        if pair in self.parents:
            return False
        if not is_root and self.max_pairs and len(self.parents) >= self.max_pairs:
            self.stats['max_pair_skips'] += 1
            return False
        self.parents[pair] = ((predecessor, selector)
                              if predecessor is not None else None)
        self.pair_trail.append(pair)
        self.by_target[q].append(pair)
        for letter in self.input.trans[p]:
            self.watch[q, letter].append(pair)
        if p in self.input.final_states:
            self.positive[q].append(pair)
        if p in self.input.reject_states:
            self.negative[q].append(pair)
        self.stats['pair_additions'] += 1
        self.stats['peak_pairs'] = max(self.stats['peak_pairs'], len(self.parents))
        todo.append(pair)
        return True

    def _drain(self, todo):
        while todo:
            pair = todo.popleft()
            p, q = divmod(pair, self.n)
            for letter, dst in self.input.trans[p].items():
                selected = self.chosen.get((q, letter))
                if selected is not None:
                    target, selector = selected
                    self._add_pair(dst, target, pair, selector, todo)
        self.stats['saturated_pairs'] += 1

    def _activate(self, literal):
        src, letter, dst = self.selectors[literal]
        self.chosen[src, letter] = (dst, literal)
        self.stats['edge_activations'] += 1
        todo = deque()
        # Snapshot length: extending reachability may append to this watch
        # list, but the new pairs are saturated separately in _drain().
        listeners = self.watch.get((src, letter), ())
        for i in range(len(listeners)):
            pair = listeners[i]
            self.stats['watcher_visits'] += 1
            p = pair // self.n
            self._add_pair(self.input.trans[p][letter], dst, pair, literal, todo)
        self._drain(todo)

    def _undo_pairs(self, trail_mark):
        while len(self.pair_trail) > trail_mark:
            pair = self.pair_trail.pop()
            p, q = divmod(pair, self.n)
            for letter in self.input.trans[p]:
                if self.watch[q, letter].pop() != pair:
                    raise AssertionError('product watch rollback is not LIFO')
            if self.by_target[q].pop() != pair:
                raise AssertionError('product target rollback is not LIFO')
            if p in self.input.final_states and self.positive[q].pop() != pair:
                raise AssertionError('positive witness rollback is not LIFO')
            if p in self.input.reject_states and self.negative[q].pop() != pair:
                raise AssertionError('negative witness rollback is not LIFO')
            del self.parents[pair]
            self.stats['pair_rollbacks'] += 1

    def on_assignment(self, lit, fixed=False):
        old = self.assignment.get(abs(lit))
        super().on_assignment(lit, fixed)
        if old == lit:
            return
        if lit > 0 and lit in self.selectors:
            if fixed and self.pair_marks:
                self._late_fixed = True
            self._activate(lit)

    def on_new_level(self):
        super().on_new_level()
        self.pair_marks.append(len(self.pair_trail))

    def on_backtrack(self, to):
        previous_level = len(self.pair_marks)
        while len(self.pair_marks) > to:
            self._undo_pairs(self.pair_marks.pop())
        super().on_backtrack(to)
        for key, (_, var) in list(self.chosen.items()):
            if self.assignment.get(var) != var:
                del self.chosen[key]
        # The IPASIR-UP fixed flag denotes a root assignment. Handle the
        # uncommon case where it arrives while the theory has higher levels.
        if previous_level > to and self._late_fixed:
            self._restore_closure()
            if to == 0:
                self._late_fixed = False

    def _restore_closure(self):
        self._drain(deque(self.parents))
        self.stats['reach_rebuilds'] += 1

    def _selected_transitions(self, assigned=None):
        if assigned is None:
            return self.chosen
        return super()._selected_transitions(assigned)

    def _support(self, pair):
        support = set()
        while self.parents[pair] is not None:
            predecessor, selector = self.parents[pair]
            support.add(selector)
            pair = predecessor
        return support

    def _lookahead_incremental(self):
        if not self.lookahead_budget:
            return None, []
        trials = 0
        for pair in reversed(self.pair_trail):
            p, src = divmod(pair, self.n)
            for letter, input_dst in self.input.trans[p].items():
                for target in range(self.n):
                    if not self.by_target[target]:
                        continue
                    selector = self._selector_id[src, letter, target]
                    if self.assignment.get(selector) == -selector:
                        continue
                    for other in self.by_target[target][:3]:
                        trials += 1
                        if self._incompatible(input_dst, other // self.n):
                            clause = self._negative_support(
                                self._support(pair) | self._support(other)
                            ) + [-selector]
                            if self.assignment.get(selector) == selector:
                                return clause, []
                            if self.assignment.get(selector) is None:
                                self.reasons[-selector] = clause
                                self.stats['excluded_edges'] += 1
                                return None, [-selector]
                        if trials >= self.lookahead_budget:
                            return None, []
        return None, []

    def _analyse_partial(self):
        # All the reachable product pairs were computed incrementally by
        # on_assignment(); labels and conflicts need only O(k) examination.
        self.stats['partial_pairs'] += len(self.parents)
        for q in range(self.n):
            plus, minus = self.positive[q], self.negative[q]
            if plus and minus:
                return self._negative_support(
                    self._support(plus[-1]) | self._support(minus[-1])), []
            witness = plus[-1] if plus else (minus[-1] if minus else None)
            if witness is None:
                continue
            conclusion = self.final[q] if plus else -self.final[q]
            clause = self._negative_support(self._support(witness)) + [conclusion]
            actual = self.assignment.get(abs(conclusion))
            if actual == -conclusion:
                return clause, []
            if actual is None:
                self.reasons[conclusion] = clause
                return None, [conclusion]
        return self._lookahead_incremental()
