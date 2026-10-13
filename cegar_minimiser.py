"""Opt-in counterexample-guided SAT minimisation for three-valued DFAs.

The original minimiser in minimiser.py is deliberately unchanged.  This module
uses the same target-DFA variables and (optionally) the same BFS and safety
constraints, but adds input-automaton product constraints only when an exact
product check discovers an incorrectly classified word.

An input state may correspond to *multiple* target states because the minimal
three-valued input automaton shares suffixes.  Product variables x[p, q] are
therefore not constrained to be one-hot.
"""

from collections import deque
import time

from pysat.formula import IDPool
from pysat.solvers import Solver

import sdfa as SDFA
from minimiser import sdfa_minimiser


class sdfa_cegar_minimiser:
    """Exact minimum consistent DFA search using lazy product constraints."""

    def __init__(self, batch_size=32, seed_per_label=4, verbose=True):
        if batch_size < 1 or seed_per_label < 0:
            raise ValueError("batch_size must be positive; seed_per_label nonnegative")
        self.batch_size = batch_size
        self.seed_per_label = seed_per_label
        self.verbose = verbose
        self.stats = []

    def _log(self, message):
        if self.verbose:
            print(message, flush=True)

    @staticmethod
    def _verify_product(input_sdfa, dfa, limit=32):
        """Return up to limit (root, word) counterexamples, and visited count.

        Missing input edges and unlabelled states are don't-cares. Explore the
        synchronous product of the whole input automaton and the candidate.
        Predecessors are retained so that actual offending words can be
        reconstructed even when the input 3DFA shares suffix states.
        """
        pred = {}
        todo = deque()
        for root in sorted(input_sdfa.init_states):
            pair = (root, 0)
            if pair not in pred:
                pred[pair] = None
                todo.append(pair)

        violations = []
        visited = 0
        while todo:
            p, q = todo.popleft()
            visited += 1
            accepting = q in dfa.final_states
            bad = ((p in input_sdfa.final_states and not accepting) or
                   (p in input_sdfa.reject_states and accepting))
            if bad:
                word = []
                at = (p, q)
                while pred[at] is not None:
                    previous, letter = pred[at]
                    word.append(letter)
                    at = previous
                word.reverse()
                violations.append((at[0], tuple(word)))
                if len(violations) >= limit:
                    return violations, visited

            for letter, dst in input_sdfa.trans[p].items():
                successor = (dst, dfa.trans[q][letter])
                if successor not in pred:
                    pred[successor] = ((p, q), letter)
                    todo.append(successor)

        return violations, visited

    @staticmethod
    def _seed_paths(input_sdfa, per_label):
        """Select short witnesses for some labelled states, without storing paths.

        Parents are retained once per 3DFA state; paths are reconstructed only
        for selected labels. This also works for a shared-suffix DAG.
        """
        if not per_label:
            return []
        need = (per_label if input_sdfa.final_states else 0,
                per_label if input_sdfa.reject_states else 0)
        todo = deque()
        parent = {}
        for root in sorted(input_sdfa.init_states):
            if root not in parent:
                parent[root] = (None, None, root)
                todo.append(root)
        found = [0, 0]
        seeds = []
        while todo and (found[0] < need[0] or found[1] < need[1]):
            p = todo.popleft()
            wanted = ((p in input_sdfa.final_states and found[0] < need[0]) or
                      (p in input_sdfa.reject_states and found[1] < need[1]))
            if wanted:
                word = []
                at = p
                while parent[at][0] is not None:
                    previous, letter, _ = parent[at]
                    word.append(letter)
                    at = previous
                word.reverse()
                seeds.append((parent[at][2], tuple(word)))
                if p in input_sdfa.final_states:
                    found[0] += 1
                if p in input_sdfa.reject_states:
                    found[1] += 1
            for letter, dst in input_sdfa.trans[p].items():
                if dst not in parent:
                    parent[dst] = (p, letter, parent[p][2])
                    todo.append(dst)
        return seeds

    @staticmethod
    def _candidate_from_model(model, edges, n, alphabet):
        positive = {literal for literal in model if literal > 0}
        result = SDFA.sdfa()
        result.set_num_states(n)
        result.set_num_letters(len(alphabet))
        result.add_initial_state(0)
        for p in range(n):
            if edges[p, -1, -1] in positive:
                result.add_final_state(p)
            for a in alphabet:
                successors = [q for q in range(n) if edges[p, a, q] in positive]
                if len(successors) != 1:
                    raise RuntimeError("SAT model is not a complete deterministic DFA")
                result.add_transition(p, a, successors[0])
        return result

    def solve(self, sat, n, alphabet, input_sdfa, nobfs, safety):
        """Return (is_sat, DFA) for a fixed target state count n."""
        start = time.perf_counter()
        pool = IDPool()
        # Reuse the original minimiser's target variable layout/conventions.
        edges = {(p, a, q): pool.id(('edge', p, a, q))
                 for p in range(n) for a in alphabet for q in range(n)}
        for p in range(n):
            edges[p, -1, -1] = pool.id(('final', p))

        active_edges = set()
        active_labels = set()
        clauses_added = 0
        sat_seconds = 0.0
        verify_seconds = 0.0
        product_visits = 0
        iterations = 0
        counterexamples = 0
        baseline = sdfa_minimiser()

        def add_clause(solver, clause):
            nonlocal clauses_added
            solver.add_clause(clause)
            clauses_added += 1

        def x(p, q):
            return pool.id(('product', p, q))

        def activate_label(solver, p):
            if p in active_labels:
                return
            active_labels.add(p)
            for q in range(n):
                if p in input_sdfa.final_states:
                    add_clause(solver, [-x(p, q), edges[q, -1, -1]])
                if p in input_sdfa.reject_states:
                    add_clause(solver, [-x(p, q), -edges[q, -1, -1]])

        def activate_path(solver, root, word):
            p = root
            activate_label(solver, p)
            for letter in word:
                dst = input_sdfa.trans[p][letter]
                if (p, letter) not in active_edges:
                    active_edges.add((p, letter))
                    for q in range(n):
                        for r in range(n):
                            add_clause(solver, [-x(p, q),
                                                -edges[q, letter, r], x(dst, r)])
                p = dst
                activate_label(solver, p)

        def finish(found, dfa):
            self.stats.append({
                'states': n,
                'sat': found,
                'iterations': iterations,
                'counterexamples': counterexamples,
                'active_edges': len(active_edges),
                'active_labels': len(active_labels),
                'clauses': clauses_added,
                'variables': pool.top,
                'product_visits': product_visits,
                'sat_seconds': round(sat_seconds, 4),
                'verify_seconds': round(verify_seconds, 4),
                'seconds': round(time.perf_counter() - start, 4),
            })
            self._log("CEGAR k={} {} after {} iterations: {} active edges, "
                      "{} clauses ({:.3f}s)".format(
                          n, 'SAT' if found else 'UNSAT', iterations,
                          len(active_edges), clauses_added,
                          time.perf_counter() - start))
            return found, dfa

        with Solver(name=sat) as solver:
            # Determinism/completeness; use each pair only once rather than the
            # duplicate ordered-pair encoding in the baseline.
            for p in range(n):
                for a in alphabet:
                    add_clause(solver, [edges[p, a, q] for q in range(n)])
                    for q in range(n):
                        for r in range(q + 1, n):
                            add_clause(solver, [-edges[p, a, q], -edges[p, a, r]])

            # These constraints are identical to the legacy minimiser's optional
            # restrictions. No changes are made to the original implementation.
            if not nobfs:
                parents = {(p, q): pool.id(('parent', p, q))
                           for p in range(n) for q in range(n)}
                t_aux = {(p, q): pool.id(('tree_edge', p, q))
                         for p in range(n) for q in range(n)}
                m_aux = {(p, a, q): pool.id(('small_letter', p, a, q))
                         for p in range(n) for a in alphabet for q in range(n)}
                for clause in baseline.create_BFStree_cnf(
                        edges, parents, t_aux, m_aux, n, alphabet, safety):
                    add_clause(solver, clause)

            if safety:
                for clause in baseline.create_DSA_cnf(edges, n, alphabet):
                    add_clause(solver, clause)

            for root in input_sdfa.init_states:
                add_clause(solver, [x(root, 0)])

            for root, word in self._seed_paths(input_sdfa, self.seed_per_label):
                activate_path(solver, root, word)

            while True:
                iterations += 1
                t = time.perf_counter()
                is_sat = solver.solve()
                sat_seconds += time.perf_counter() - t
                if not is_sat:
                    return finish(False, None)

                dfa = self._candidate_from_model(
                    solver.get_model(), edges, n, alphabet)
                t = time.perf_counter()
                witnesses, visits = self._verify_product(
                    input_sdfa, dfa, self.batch_size)
                verify_seconds += time.perf_counter() - t
                product_visits += visits
                if not witnesses:
                    return finish(True, dfa)

                counterexamples += len(witnesses)
                old_size = (len(active_edges), len(active_labels))
                for root, word in witnesses:
                    activate_path(solver, root, word)
                if old_size == (len(active_edges), len(active_labels)):
                    raise RuntimeError(
                        "A violating path is already fully encoded; "
                        "check the verifier and SAT encoding")

    def minimise(self, input_sdfa, sat, lbound, ubound, nobfs, safety):
        """Find the smallest allowed DFA, as in the original minimiser."""
        self.stats = []
        alphabet = list(range(input_sdfa.num_letters))
        max_bound = min(input_sdfa.num_states, ubound)
        self._log("Input SDFA size: {}, alphabet size: {}".format(
            input_sdfa.num_states, len(alphabet)))
        for n in range(max(1, lbound), max_bound + 1):
            self._log("CEGAR: looking for a DFA with {} states...".format(n))
            found, result = self.solve(
                sat, n, alphabet, input_sdfa, nobfs, safety)
            if found:
                return result
        return None
