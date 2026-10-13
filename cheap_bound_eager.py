"""Exact eager minimisation with a cheap, certified residual lower bound."""
from eager_cnf_variants import ReducedEagerMinimiser
from cheap_lower_bound import CheapIncompatibilityBound


class CheapBoundEagerMinimiser(ReducedEagerMinimiser):
    def __init__(self, budget_ms=25, max_suffixes=32, max_candidates=256,
                 max_depth=8, verbose=True, pair_budget=300,
                 pair_budget_ms=15, pair_policy='impact'):
        super().__init__('ordered')
        self.budget_ms = budget_ms
        self.max_suffixes = max_suffixes
        self.max_candidates = max_candidates
        self.max_depth = max_depth
        self.verbose = verbose
        self.certificate = None
        self.pair_budget = pair_budget
        self.pair_budget_ms = pair_budget_ms
        self.pair_policy = pair_policy
        self.selected_pairs = []

    def create_cnf(self, nodes, edges, parents, t_aux, m_aux,
                   input_sdfa, n, alphabet, nobfs, safety):
        clauses = super().create_cnf(nodes, edges, parents, t_aux, m_aux,
                                      input_sdfa, n, alphabet, nobfs, safety)
        for p, q, witness in self.selected_pairs:
            for i in range(n):
                clauses.append([-nodes[p, i], -nodes[q, i]])
        return clauses

    def minimise(self, input_sdfa, sat, lbound, ubound, nobfs, safety):
        # The bound applies to any consistent complete target DFA; in safety
        # mode we still use only the ordinary consistency requirement.
        bound = CheapIncompatibilityBound(
            input_sdfa, max_suffixes=self.max_suffixes,
            max_candidates=self.max_candidates,
            max_depth=self.max_depth, budget_ms=self.budget_ms)
        if not bound.verify():
            raise AssertionError('invalid residual lower-bound certificate')
        self.certificate = bound
        self.selected_pairs = bound.scored_pairs(
            limit=self.pair_budget, budget_ms=self.pair_budget_ms,
            policy=self.pair_policy)
        bound.verify_scored_pairs(self.selected_pairs)
        if self.verbose:
            print('Selected witnessed incompatibility pairs: {}'.format(len(self.selected_pairs)))
        if self.verbose:
            print('Cheap incompatibility lower bound: {} ({} suffixes, {:.2f} ms)'.format(
                bound.lower_bound, len(bound.suffixes), bound.elapsed_ms))
        return super().minimise(input_sdfa, sat, max(lbound, bound.lower_bound),
                                ubound, nobfs, safety)
