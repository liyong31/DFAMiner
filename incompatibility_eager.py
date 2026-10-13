"""Opt-in eager SAT with certified clique lower bounds and selected pair clauses."""
from eager_cnf_variants import ReducedEagerMinimiser


class IncompatibilityEagerMinimiser(ReducedEagerMinimiser):
    def __init__(self, mode='ordered', pair_budget=0, verbose=True):
        super().__init__(mode)
        self.pair_budget = pair_budget
        self.verbose = verbose
        self.last_lower_bound = 1

    def create_cnf(self, nodes, edges, parents, t_aux, m_aux,
                   input_sdfa, n, alphabet, nobfs, safety):
        clauses = super().create_cnf(nodes, edges, parents, t_aux, m_aux,
                                     input_sdfa, n, alphabet, nobfs, safety)
        info = getattr(input_sdfa, 'incompatibility', None)
        if info is not None:
            for p,q in info.selected_pairs(self.pair_budget):
                for state in range(n):
                    clauses.append([-nodes[p,state],-nodes[q,state]])
        return clauses

    def minimise(self, input_sdfa, sat, lbound, ubound, nobfs, safety):
        info = getattr(input_sdfa, 'incompatibility', None)
        if info is not None and not safety:
            # Clique states are reachable and pairwise distinguishable,
            # hence every consistent target requires >= clique size.
            self.last_lower_bound = max(lbound, len(info.clique), 1)
        else:
            self.last_lower_bound = lbound
        if self.verbose:
            print('Incompatibility clique lower bound:', self.last_lower_bound)
        return super().minimise(input_sdfa, sat, self.last_lower_bound,
                                ubound, nobfs, safety)
