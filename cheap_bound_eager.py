"""Exact eager minimisation with a cheap, certified residual lower bound."""
from eager_cnf_variants import ReducedEagerMinimiser
from cheap_lower_bound import CheapIncompatibilityBound


class CheapBoundEagerMinimiser(ReducedEagerMinimiser):
    def __init__(self, budget_ms=25, max_suffixes=32, max_candidates=256,
                 max_depth=8, verbose=True):
        super().__init__('ordered')
        self.budget_ms = budget_ms
        self.max_suffixes = max_suffixes
        self.max_candidates = max_candidates
        self.max_depth = max_depth
        self.verbose = verbose
        self.certificate = None

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
        if self.verbose:
            print('Cheap incompatibility lower bound: {} ({} suffixes, {:.2f} ms)'.format(
                bound.lower_bound, len(bound.suffixes), bound.elapsed_ms))
        return super().minimise(input_sdfa, sat, max(lbound, bound.lower_bound),
                                ubound, nobfs, safety)
