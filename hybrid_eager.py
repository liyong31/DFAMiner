"""Opt-in hybrid eager SAT for minimum consistent DFA inference.

The target transition/product formula stays eager.  For large 3DFAs we
omit the explicit quadratic opposite-labelled residual-pair clauses; they
are implied by the accepting/rejecting-state implications already present.

We combine:
  1. the existing target-BFS symmetry-breaking clauses;
  2. the compact 'both' eager CNF;
  3. up to N certified, scored incompatibility-pair clauses;
  4. a cheap witnessed clique lower bound on the target DFA size.

No 3DFA-to-NFA relaxation, theory callbacks or CEGAR is used.  This
module deliberately keeps the optimisation opt-in until native benchmarks
show that the combination beats compact eager with BFS alone.
"""
from time import perf_counter

from cheap_lower_bound import CheapIncompatibilityBound
from eager_cnf_variants import ReducedEagerMinimiser


class HybridEagerMinimiser(ReducedEagerMinimiser):
    def __init__(self, pair_budget=300, policy='short', bound_budget_ms=25,
                 pair_budget_ms=15, max_candidates=256, max_suffixes=64,
                 max_depth=9, explicit_label_pairs=False,
                 verbose=False):
        # mode='both': ordered transition clauses, omit implied F x R
        # mode='ordered': ordered transition clauses, keep implied F x R
        super().__init__('ordered' if explicit_label_pairs else 'both')
        self.pair_budget = pair_budget
        self.policy = policy
        self.bound_budget_ms = bound_budget_ms
        self.pair_budget_ms = pair_budget_ms
        self.max_candidates = max_candidates
        self.max_suffixes = max_suffixes
        self.max_depth = max_depth
        self.verbose = verbose
        self.selected_pairs = ()
        self.certificate = None
        self.preparation_seconds = 0.0
        self.pair_generation_seconds = 0.0

    def prepare(self, machine):
        """Compute bounded witnesses once and check every selected clause.

        The bound is useful for a full increasing-k minimisation, but its
        addition never eliminates target sizes >= the true optimum.
        """
        self.selected_pairs = ()
        self.certificate = None
        start = perf_counter()
        if self.bound_budget_ms > 0:
            self.certificate = CheapIncompatibilityBound(
                machine, budget_ms=self.bound_budget_ms,
                max_candidates=self.max_candidates,
                max_suffixes=self.max_suffixes,
                max_depth=self.max_depth)
            self.certificate.verify()
        bound_finish = perf_counter()
        if self.certificate and self.pair_budget:
            pairs = self.certificate.scored_pairs(
                limit=self.pair_budget,
                budget_ms=self.pair_budget_ms,
                policy=self.policy)
            self.certificate.verify_scored_pairs(pairs)
            self.selected_pairs = tuple(pairs)
        self.pair_generation_seconds = perf_counter() - bound_finish
        self.preparation_seconds = perf_counter() - start
        if self.verbose:
            print('Hybrid eager: lower bound = {}, selected pairs = {}, prep = {:.3f}s'.format(
                self.lower_bound,len(self.selected_pairs),self.preparation_seconds))
        return self

    @property
    def lower_bound(self):
        return self.certificate.lower_bound if self.certificate else 1

    def create_cnf(self, nodes, edges, parents, t_aux, m_aux,
                   input_sdfa, n, alphabet, nobfs, safety):
        clauses = super().create_cnf(nodes, edges, parents, t_aux, m_aux,
                                     input_sdfa, n, alphabet, nobfs, safety)
        for p, q, _suffix in self.selected_pairs:
            for i in range(n):
                clauses.append([-nodes[p, i], -nodes[q, i]])
        return clauses

    def minimise(self, input_sdfa, sat, lbound, ubound, nobfs, safety):
        self.prepare(input_sdfa)
        return super().minimise(input_sdfa, sat,
                                max(lbound, self.lower_bound),
                                ubound, nobfs, safety)
