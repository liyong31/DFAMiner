"""Repeatable native CaDiCaL comparison on generated DFA-learning workloads.

All algorithms solve exactly the same 3DFA for each k. Tests include hard
UNSAT sizes below the optimum as well as SAT size. No sample enumeration
occurs during minimisation. Times include CNF construction and model checking.
"""
import argparse
import itertools
import json
import random
import statistics
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pysat.solvers import Solver
from sdfa import sdfa
from minimiser import sdfa_minimiser
from cegar_minimiser import sdfa_cegar_minimiser
from cdcl_product import CDCLProductMinimiser


def compressed_trie(alphabet, words):
    """Minimise an acyclic three-valued trie by bottom-up hash consing."""
    transitions = [{}]
    labels = [0]
    for word, value in words:
        state = 0
        for a in word:
            nxt = transitions[state].get(a)
            if nxt is None:
                nxt = len(transitions)
                transitions[state][a] = nxt
                transitions.append({})
                labels.append(0)
            state = nxt
        if labels[state] not in (0, value):
            raise AssertionError('contradictory sample')
        labels[state] = value

    intern = {}
    new_trans, new_labels = [], []

    def minimise(state):
        successors = tuple(sorted((a, minimise(dst))
                                  for a, dst in transitions[state].items()))
        signature = (labels[state], successors)
        if signature not in intern:
            q = len(new_trans)
            intern[signature] = q
            new_trans.append(dict(successors))
            new_labels.append(labels[state])
        return intern[signature]

    root = minimise(0)
    m = sdfa()
    m.set_num_states(len(new_trans))
    m.set_num_letters(alphabet)
    m.add_initial_state(root)
    for p, outgoing in enumerate(new_trans):
        for a, q in outgoing.items():
            m.add_transition(p, a, q)
        if new_labels[p] == 1:
            m.add_final_state(p)
        if new_labels[p] == -1:
            m.add_reject_state(p)
    return m


def parity_label(word):
    priorities = []
    for i in range(len(word)):
        for j in range(i + 1, len(word)):
            if word[i] == word[j]:
                priorities.append(max(word[i:j+1]) & 1)
    if not priorities:
        return 0
    if all(p == 0 for p in priorities):
        return 1
    if all(p == 1 for p in priorities):
        return -1
    return 0


def generate(case):
    if case.startswith('priority'):
        alphabet, depth = map(int, case[len('priority'):].split('x'))
        words = ((w, parity_label(w))
                 for length in range(1, depth + 1)
                 for w in itertools.product(range(alphabet), repeat=length))
        return compressed_trie(alphabet, ((w,v) for w,v in words if v))
    if case.startswith('mod'):
        parts = case[len('mod'):].split('x')
        modulus, depth = map(int, parts)
        words = ((w, 1 if sum(1 for a in w if a == 1) % modulus == 0 else -1)
                 for length in range(depth + 1)
                 for w in itertools.product(range(2), repeat=length))
        return compressed_trie(2, words)
    if case.startswith('random'):
        # Reproducible random DFA of size k: classify all short binary words.
        k, depth, seed = map(int, case[len('random'):].split('x'))
        rng = random.Random(seed)
        table = [[rng.randrange(k) for _ in range(2)] for _ in range(k)]
        accepting = {q for q in range(k) if rng.randrange(2)}
        def classification(word):
            q = 0
            for a in word:
                q = table[q][a]
            return 1 if q in accepting else -1
        words = ((w, classification(w))
                 for length in range(depth + 1)
                 for w in itertools.product(range(2), repeat=length))
        return compressed_trie(2, words)
    raise ValueError(case)


def solve_at(m, k, mode, bfs=False, frequency=4, lookahead=32, pairs=12000):
    alpha = list(range(m.num_letters))
    start = time.perf_counter()
    if mode == 'eager':
        base = sdfa_minimiser()
        nodes, edges, parents, t_aux, m_aux = base.create_variables(k, alpha, m.num_states)
        clauses = base.create_cnf(nodes, edges, parents, t_aux, m_aux,
                                  m, k, alpha, not bfs, False)
        with Solver(name='cadical195', bootstrap_with=clauses) as s:
            sat = s.solve()
            details = {'clauses': len(clauses), 'solver': s.accum_stats()}
            dfa = base.construct_dfa_from_model(s.get_model(), edges, k, alpha) if sat else None
    elif mode == 'cegar':
        obj = sdfa_cegar_minimiser(verbose=False)
        sat, dfa = obj.solve('cadical195', k, alpha, m, not bfs, False)
        details = obj.stats[-1]
    else:
        obj = CDCLProductMinimiser(check_frequency=frequency,
                                  lookahead_budget=lookahead,
                                  max_pairs=pairs,
                                  verbose=False)
        sat, dfa = obj.solve('cadical195', k, alpha, m, not bfs, False)
        details = obj.stats[-1]
    wall = time.perf_counter() - start
    if sat:
        errors, visits = sdfa_cegar_minimiser._verify_product(m, dfa, limit=1)
        if errors:
            raise AssertionError((mode, k, errors))
    return {'mode':mode,'k':k,'sat':sat,'seconds':round(wall,5),'details':details}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cases', nargs='+', default=[
        'mod3x7','mod4x8','priority3x6','priority3x7','random5x8x17'])
    p.add_argument('--ks', nargs='+', type=int, default=[2,3,4])
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--modes', nargs='+', default=['eager','cegar','cdcl'])
    p.add_argument('--frequency', type=int, default=4)
    p.add_argument('--lookahead', type=int, default=32)
    p.add_argument('--pairs', type=int, default=12000)
    p.add_argument('--bfs', action='store_true')
    p.add_argument('--output', default='benchmark_cdcl_hard_results.json')
    args = p.parse_args()
    results = []
    for case in args.cases:
        m = generate(case)
        print('CASE',case,'3dfa_states',m.num_states,
              'edges',sum(map(len,m.trans)),
              'pos',len(m.final_states),'neg',len(m.reject_states),flush=True)
        for k in args.ks:
            if k > m.num_states:
                continue
            for mode in args.modes:
                for repeat in range(args.repeats):
                    row = solve_at(m,k,mode,args.bfs,args.frequency,args.lookahead,args.pairs)
                    row.update(case=case,repeat=repeat,input_states=m.num_states)
                    print(json.dumps(row),flush=True)
                    results.append(row)
            statuses = {v['sat'] for v in results if v['case']==case and v['k']==k}
            if len(statuses) != 1:
                raise AssertionError((case,k,statuses))
    Path(args.output).write_text(json.dumps(results, indent=2))
    print('COMPLETED',len(results),'measurements',flush=True)


if __name__ == '__main__':
    main()
