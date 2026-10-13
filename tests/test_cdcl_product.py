"""Theory-level soundness checks for the CDCL product propagator."""
import itertools
import random
import unittest
from pathlib import Path
from cdcl_product import DFAProductPropagator, CDCLProductMinimiser
from cegar_minimiser import sdfa_cegar_minimiser
from sdfa import sdfa


def input_machine(seed):
    rng = random.Random(seed)
    machine = sdfa()
    machine.set_num_states(rng.randrange(1, 6))
    machine.set_num_letters(2)
    machine.add_initial_state(0)
    for p in range(machine.num_states):
        value = rng.randrange(4)
        if value == 1: machine.add_final_state(p)
        if value == 2: machine.add_reject_state(p)
        if value == 3:
            machine.add_final_state(p)
            machine.add_reject_state(p)
        for a in range(2):
            if rng.randrange(3):
                machine.add_transition(p, a, rng.randrange(machine.num_states))
    return machine


def encoding(k):
    edges = {}
    for p in range(k):
        for a in range(2):
            for q in range(k):
                edges[p, a, q] = len(edges) + 1
    for p in range(k):
        edges[p, -1, -1] = len(edges) + 1
    return edges


def all_models(k, edges):
    for choices in itertools.product(range(k), repeat=2*k):
        for bitmask in range(1 << k):
            transitions = {(p, a): choices[2*p+a] for p in range(k) for a in range(2)}
            finals = {p for p in range(k) if bitmask & (1 << p)}
            literals = set()
            for (p, a), target in transitions.items():
                for q in range(k):
                    v = edges[p, a, q]
                    literals.add(v if q == target else -v)
            for p in range(k):
                v = edges[p, -1, -1]
                literals.add(v if p in finals else -v)
            yield transitions, finals, literals


def consistent(machine, transitions, finals):
    todo = [(p, 0) for p in machine.init_states]
    reached = set(todo)
    for p, q in todo:
        if p in machine.final_states and q not in finals: return False
        if p in machine.reject_states and q in finals: return False
        for a, dst in machine.trans[p].items():
            pair = (dst, transitions[q, a])
            if pair not in reached:
                reached.add(pair)
                todo.append(pair)
    return True


class ProductTheoryTests(unittest.TestCase):
    def test_every_model_rejection_lemma_is_valid(self):
        # Exhaustively check all two-state target DFAs on 24 small instances.
        for seed in range(24):
            machine, edges = input_machine(seed), encoding(2)
            models = list(all_models(2, edges))
            valid = [m for m in models if consistent(machine, m[0], m[1])]
            for transitions, finals, literals in models:
                theory = DFAProductPropagator(machine, 2, edges)
                theory_valid = theory.check_model(list(literals))
                self.assertEqual(theory_valid, consistent(machine, transitions, finals))
                if not theory_valid:
                    clause = theory.add_clause()
                    self.assertFalse(any(l in literals for l in clause))
                    for _, _, assignment in valid:
                        self.assertTrue(any(l in assignment for l in clause))

    def test_partial_explanations_are_valid(self):
        for seed in range(35):
            machine, edges = input_machine(seed + 300), encoding(2)
            valid = [m for m in all_models(2, edges) if consistent(machine, m[0], m[1])]
            theory = DFAProductPropagator(machine, 2, edges, check_frequency=1)
            theory.on_assignment(edges[0, 0, seed % 2])
            conflict, inferred = theory._analyse_partial()
            clauses = ([] if conflict is None else [conflict])
            clauses += [theory.provide_reason(l) for l in inferred]
            for clause in clauses:
                for _, _, assignment in valid:
                    self.assertTrue(any(l in assignment for l in clause))

    def test_backtracking_removes_partial_assignments(self):
        machine, edges = input_machine(9), encoding(2)
        theory = DFAProductPropagator(machine, 2, edges)
        variable = edges[0, 0, 0]
        theory.on_new_level()
        theory.on_assignment(variable)
        self.assertIn(variable, theory.assignment)
        theory.on_backtrack(0)
        self.assertNotIn(variable, theory.assignment)

    def test_opposite_initial_labels(self):
        machine = sdfa()
        machine.set_num_states(1)
        machine.set_num_letters(2)
        machine.add_initial_state(0)
        machine.add_final_state(0)
        machine.add_reject_state(0)
        theory = DFAProductPropagator(machine, 1, encoding(1))
        conflict, _ = theory._analyse_partial()
        self.assertEqual(conflict, [])

    def test_native_integration(self):
        # Local model-substitute environments do not expose IPASIR-UP.
        from pysat.solvers import Solver
        if not hasattr(Solver, 'connect_propagator'):
            self.skipTest('native PySAT and CaDiCaL 1.9.5 unavailable')
        machine = sdfa()
        machine.load(str(Path(__file__).resolve().parents[1] / 'intermediate_3dfa.txt'))
        minimiser = CDCLProductMinimiser(check_frequency=1, verbose=False)
        result = minimiser.minimise(machine, 'cadical195', 1, 8, True, False)
        self.assertIsNotNone(result)
        self.assertEqual(result.num_states, 3)
        self.assertFalse(sdfa_cegar_minimiser._verify_product(machine, result)[0])


if __name__ == '__main__':
    unittest.main()
