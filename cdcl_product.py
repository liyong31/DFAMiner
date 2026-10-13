"""Exact DFA inference with a CaDiCaL IPASIR-UP product propagator.

This module is opt-in.  In particular, the minimal 3DFA's suffix-shared states
are never constrained to map to exactly one target DFA state.

It implements three sound CDCL theory inferences:
- selected target edges reaching a labelled input state require its label;
- opposite input labels reaching one target state give a conflict clause;
- a distinguishing suffix can rule out a candidate target transition.

Every inferred literal has an implication clause and every conflict has a
clause implied by the original labelled-word separation problem.  Full model
checking remains exact even with bounded partial propagation.
"""

from collections import deque
from time import perf_counter

from pysat.engines import Propagator
from pysat.formula import IDPool
from pysat.solvers import Solver

import sdfa as SDFA
from minimiser import sdfa_minimiser
from cegar_minimiser import sdfa_cegar_minimiser


class DFAProductPropagator(Propagator):
    """Provide rollbackable explanations of forced product reachability."""

    def __init__(self, input_sdfa, n, edges, max_pairs=12000,
                 check_frequency=4, lookahead_budget=32,
                 suffix_budget=1000):
        if (check_frequency < 1 or max_pairs < 0 or
                lookahead_budget < 0 or suffix_budget < 1):
            raise ValueError('invalid CDCL theory propagation budget')
        self.input = input_sdfa
        self.n = n
        self.max_pairs = max_pairs
        self.check_frequency = check_frequency
        self.lookahead_budget = lookahead_budget
        self.suffix_budget = suffix_budget
        self.selectors = {lit: (p, a, q) for (p, a, q), lit in edges.items()
                          if a != -1}
        self._selector_id = {key: lit for key, lit in edges.items()
                             if key[1] != -1}
        self.final = {p: edges[p, -1, -1] for p in range(n)}
        self.final_vars = set(self.final.values())
        self.observed = sorted(set(self.selectors) | self.final_vars)

        self.assignment = {}
        self.fixed = set()
        self.trail = []
        self.trlim = []
        self.reasons = {}
        self._clause = None
        self._incompat_cache = {}
        self._dirty = True
        self._changes = check_frequency

        self.stats = dict(partial_checks=0, partial_pairs=0,
                          theory_propagations=0, early_conflicts=0,
                          model_checks=0, rejected_models=0,
                          added_clauses=0, backtracks=0,
                          excluded_edges=0, suffix_probes=0)

    def setup_observe(self, solver):
        self.solver = solver
        for var in self.observed:
            solver.observe(var)

    def on_assignment(self, lit, fixed=False):
        var = abs(lit)
        if var not in self.final_vars and var not in self.selectors:
            raise AssertionError('unexpected unobserved assignment')
        if self.assignment.get(var) == lit:
            return
        self.assignment[var] = lit
        self.trail.append(lit)
        if fixed:
            self.fixed.add(var)
        if lit > 0 and var in self.selectors:
            self._changes += 1
            self._dirty = True
        elif var in self.final_vars:
            self._changes = self.check_frequency
            self._dirty = True

    def on_new_level(self):
        self.trlim.append(len(self.trail))

    def on_backtrack(self, to):
        self.stats['backtracks'] += 1
        while len(self.trlim) > to:
            start = self.trlim.pop()
            while len(self.trail) > start:
                lit = self.trail.pop()
                if abs(lit) not in self.fixed:
                    self.assignment.pop(abs(lit), None)
        for lit in list(self.reasons):
            if self.assignment.get(abs(lit)) != lit:
                del self.reasons[lit]
        self._changes = self.check_frequency
        self._dirty = True

    @staticmethod
    def _path_support(pred, node):
        used = set()
        while pred[node] is not None:
            predecessor, selector = pred[node]
            used.add(selector)
            node = predecessor
        return used

    @staticmethod
    def _negative_support(support):
        return [-var for var in sorted(support)]

    def _selected_transitions(self, assigned=None):
        if assigned is None:
            assigned = self.assignment
        selected = {}
        for var, lit in assigned.items():
            if lit <= 0 or var not in self.selectors:
                continue
            p, a, q = self.selectors[var]
            selected[p, a] = (q, var)
        return selected

    def _incompatible(self, start_p, start_q):
        """True only with a proved opposite-label common suffix.

        A bounded search failure is treated as unknown, never as evidence
        of compatibility.  Instance-local caching does not retain old 3DFAs.
        """
        key = (min(start_p, start_q), max(start_p, start_q))
        if key in self._incompat_cache:
            return self._incompat_cache[key]
        if len(self._incompat_cache) >= 32768:
            self._incompat_cache.clear()
        todo = deque([(start_p, start_q)])
        seen = {(start_p, start_q)}
        while todo and len(seen) <= self.suffix_budget:
            p, q = todo.popleft()
            self.stats['suffix_probes'] += 1
            if ((p in self.input.final_states and q in self.input.reject_states)
                or (p in self.input.reject_states and q in self.input.final_states)):
                self._incompat_cache[key] = True
                return True
            tr1, tr2 = self.input.trans[p], self.input.trans[q]
            for a, r in tr1.items():
                s = tr2.get(a)
                if s is not None and (r, s) not in seen:
                    seen.add((r, s))
                    todo.append((r, s))
        self._incompat_cache[key] = False
        return False

    def _lookahead(self, pred):
        """Rule out a transition that would merge incompatible residuals."""
        if not self.lookahead_budget:
            return None, []
        by_target = {}
        for p, i in pred:
            if len(by_target.get(i, ())) < 3:
                by_target.setdefault(i, []).append((p, i))
        trials = 0
        for p, i in pred:
            for a, r in self.input.trans[p].items():
                for j, witnesses in by_target.items():
                    lit = self._selector_id[i, a, j]
                    if self.assignment.get(lit) == -lit:
                        continue
                    for qpair in witnesses:
                        trials += 1
                        if self._incompatible(r, qpair[0]):
                            support = (self._path_support(pred, (p, i)) |
                                       self._path_support(pred, qpair))
                            clause = self._negative_support(support) + [-lit]
                            if self.assignment.get(lit) == lit:
                                return clause, []
                            if self.assignment.get(lit) is None:
                                self.reasons.setdefault(-lit, clause)
                                self.stats['excluded_edges'] += 1
                                return None, [-lit]
                        if trials >= self.lookahead_budget:
                            return None, []
        return None, []

    def _analyse_partial(self):
        """Propagate consequences of already selected target transitions.

        Rebuild reachability after backtracking rather than maintaining a
        rollbackable graph.  The pair bound limits per-callback overhead.
        """
        chosen = self._selected_transitions()
        pred = {}
        todo = deque()
        for root in sorted(self.input.init_states):
            pair = (root, 0)
            if pair not in pred:
                pred[pair] = None
                todo.append(pair)

        requirements = {}
        candidate = {}
        visited = 0
        while todo and (not self.max_pairs or visited < self.max_pairs):
            p, q = todo.popleft()
            visited += 1
            if p in self.input.final_states or p in self.input.reject_states:
                signs = []
                if p in self.input.final_states:
                    signs.append(True)
                if p in self.input.reject_states:
                    signs.append(False)
                for positive in signs:
                    prior = requirements.get(q)
                    if prior is not None and prior[0] != positive:
                        both = (self._path_support(pred, prior[1]) |
                                self._path_support(pred, (p, q)))
                        self.stats['partial_pairs'] += visited
                        return self._negative_support(both), []
                    requirements.setdefault(q, (positive, (p, q)))
                    required = self.final[q] if positive else -self.final[q]
                    reason = (self._negative_support(
                              self._path_support(pred, (p, q))) + [required])
                    current = self.assignment.get(abs(required))
                    if current == -required:
                        self.stats['partial_pairs'] += visited
                        return reason, []
                    if current is None:
                        candidate.setdefault(required, reason)

            for a, dst in self.input.trans[p].items():
                transition = chosen.get((q, a))
                if transition is None:
                    continue
                to_state, selector = transition
                pair = (dst, to_state)
                if pair not in pred:
                    pred[pair] = ((p, q), selector)
                    todo.append(pair)

        self.stats['partial_pairs'] += visited
        conflict, forbidden = self._lookahead(pred)
        if conflict is not None or forbidden:
            return conflict, forbidden
        for lit, clause in candidate.items():
            self.reasons.setdefault(lit, clause)
            return None, [lit]
        return None, []

    def propagate(self):
        if self._clause is not None or not self._dirty:
            return []
        if self._changes < self.check_frequency:
            return []
        self._changes = 0
        self._dirty = False
        self.stats['partial_checks'] += 1
        conflict, inferred = self._analyse_partial()
        if conflict is not None:
            self._clause = conflict
            self.stats['early_conflicts'] += 1
            return []
        self.stats['theory_propagations'] += len(inferred)
        return inferred

    def provide_reason(self, lit):
        clause = self.reasons[lit]
        if lit not in clause:
            raise AssertionError('theory reason omits conclusion')
        return clause

    def has_clause(self):
        return self._clause is not None

    def add_clause(self):
        if self._clause is None:
            raise RuntimeError('no external clause is pending')
        result = self._clause
        self._clause = None
        self.stats['added_clauses'] += 1
        return result

    def check_model(self, model):
        """Independently verify every complete target assignment."""
        self.stats['model_checks'] += 1
        positive = {v for v in model if v > 0}
        candidate = SDFA.sdfa()
        candidate.set_num_states(self.n)
        candidate.set_num_letters(self.input.num_letters)
        candidate.add_initial_state(0)
        chosen = self._selected_transitions({v: v for v in positive})
        for p in range(self.n):
            if self.final[p] in positive:
                candidate.add_final_state(p)
            for a in range(self.input.num_letters):
                transition = chosen.get((p, a))
                if transition is None:
                    raise RuntimeError('complete SAT model has undefined DFA edge')
                candidate.add_transition(p, a, transition[0])
        self.candidate = candidate
        violations, _ = sdfa_cegar_minimiser._verify_product(
            self.input, candidate, limit=1)
        if not violations:
            return True

        root, word = violations[0]
        state_m, state_d = root, 0
        support = set()
        for a in word:
            dst = candidate.trans[state_d][a]
            support.add(self._selector_id[(state_d, a, dst)])
            state_m = self.input.trans[state_m][a]
            state_d = dst
        desired = (self.final[state_d]
                   if (state_m in self.input.final_states and
                       state_d not in candidate.final_states)
                   else -self.final[state_d])
        self._clause = self._negative_support(support) + [desired]
        self.stats['rejected_models'] += 1
        return False

    def decide(self):
        return 0


class CDCLProductMinimiser:
    """Exact minimum-DFA synthesis using an IPASIR-UP user propagator."""

    def __init__(self, max_pairs=12000, check_frequency=4,
                 lookahead_budget=32, suffix_budget=1000, verbose=True):
        self.max_pairs = max_pairs
        self.check_frequency = check_frequency
        self.lookahead_budget = lookahead_budget
        self.suffix_budget = suffix_budget
        self.verbose = verbose
        self.stats = []

    def solve(self, sat, n, alphabet, input_sdfa, nobfs, safety):
        if sat != 'cadical195':
            raise ValueError('CDCL product propagation requires --solver cadical195')
        start = perf_counter()
        pool = IDPool()
        edges = {(p, a, q): pool.id(('edge', p, a, q))
                 for p in range(n) for a in alphabet for q in range(n)}
        for p in range(n):
            edges[p, -1, -1] = pool.id(('final', p))

        parents, t_aux, m_aux = {}, {}, {}
        if not nobfs:
            parents = {(p, q): pool.id(('parent', p, q))
                       for p in range(n) for q in range(n)}
            t_aux = {(p, q): pool.id(('tree_edge', p, q))
                     for p in range(n) for q in range(n)}
            m_aux = {(p, a, q): pool.id(('small_letter', p, a, q))
                     for p in range(n) for a in alphabet for q in range(n)}

        engine = DFAProductPropagator(input_sdfa, n, edges,
                                      max_pairs=self.max_pairs,
                                      check_frequency=self.check_frequency,
                                      lookahead_budget=self.lookahead_budget,
                                      suffix_budget=self.suffix_budget)
        structural_clauses = 0
        with Solver(name=sat) as solver:
            for p in range(n):
                for a in alphabet:
                    solver.add_clause([edges[p, a, q] for q in range(n)])
                    structural_clauses += 1
                    for q in range(n):
                        for r in range(q + 1, n):
                            solver.add_clause([-edges[p, a, q], -edges[p, a, r]])
                            structural_clauses += 1
            baseline = sdfa_minimiser()
            if not nobfs:
                for clause in baseline.create_BFStree_cnf(
                        edges, parents, t_aux, m_aux, n, alphabet, safety):
                    solver.add_clause(clause)
                    structural_clauses += 1
            if safety:
                for clause in baseline.create_DSA_cnf(edges, n, alphabet):
                    solver.add_clause(clause)
                    structural_clauses += 1

            solver.connect_propagator(engine)
            engine.setup_observe(solver)
            found = solver.solve()
            dfa = engine.candidate if found else None
            record = dict(states=n, sat=found,
                          seconds=round(perf_counter() - start, 4),
                          structural_clauses=structural_clauses,
                          **engine.stats)
            self.stats.append(record)
            if self.verbose:
                print('CDCL-product k={} {} {:.3f}s; {}'.format(
                    n, 'SAT' if found else 'UNSAT', record['seconds'],
                    engine.stats), flush=True)
            return found, dfa

    def minimise(self, input_sdfa, sat, lbound, ubound, nobfs, safety):
        self.stats = []
        for k in range(max(1, lbound),
                       min(input_sdfa.num_states, ubound) + 1):
            found, candidate = self.solve(
                sat, k, list(range(input_sdfa.num_letters)),
                input_sdfa, nobfs, safety)
            if found:
                return candidate
        return None
