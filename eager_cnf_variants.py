"""Controlled SAT-CNF redundancy experiments for DFAMiner eager encoding.

Variants retain the original eager encoding and variable numbering by default.
- ordered: eliminate the duplicate direction in exactly-one transition clauses.
- implied: omit opposite-labelled residual pair clauses, which are entailed by
  positive/negative label -> target finality implications.
- both: combine the two.
- dedup: both plus canonical duplicate/tautology cleanup.

Pair clauses are redundant because (-x[p,i] or final[i]) and
(-x[q,i] or -final[i]) resolve to (-x[p,i] or -x[q,i]).
Correctness of the elimination holds even with shared suffixes and cycles.
"""

from minimiser import sdfa_minimiser


class ReducedEagerMinimiser(sdfa_minimiser):
    def __init__(self, mode='both'):
        if mode not in ('baseline','ordered','implied','both','dedup'):
            raise ValueError(mode)
        self.mode = mode
        self.counts = {}

    def create_dfa_cnf(self, nodes, edges, input_sdfa, n, alphabet):
        if self.mode == 'baseline':
            return super().create_dfa_cnf(nodes, edges, input_sdfa, n, alphabet)
        out = []
        append = out.append
        for p in range(n):
            for a in alphabet:
                append([edges[p, a, q] for q in range(n)])
                for q in range(n):
                    for qp in range(n):
                        if q != qp and (self.mode not in ('ordered','both','dedup') or q < qp):
                            append([-edges[p, a, q], -edges[p, a, qp]])
        for root in input_sdfa.init_states:
            append([nodes[root, 0]])
            for i in range(1, n):
                append([-nodes[root, i]])
        for p in input_sdfa.final_states:
            for i in range(n):
                append([-nodes[p, i], edges[i, -1, -1]])
        for p in input_sdfa.reject_states:
            for i in range(n):
                append([-nodes[p, i], -edges[i, -1, -1]])
        for p, outgoing in enumerate(input_sdfa.trans):
            for a, dst in outgoing.items():
                for i in range(n):
                    for j in range(n):
                        append([-nodes[p, i], -edges[i, a, j], nodes[dst, j]])
        if self.mode not in ('implied','both','dedup'):
            for p in input_sdfa.final_states:
                for q in input_sdfa.reject_states:
                    for i in range(n):
                        append([-nodes[p, i], -nodes[q, i]])
        return out

    def create_cnf(self, nodes, edges, parents, t_aux, m_aux,
                   input_sdfa, n, alphabet, nobfs, safety):
        result = super().create_cnf(
            nodes, edges, parents, t_aux, m_aux, input_sdfa,
            n, alphabet, nobfs, safety)
        self.counts['raw'] = len(result)
        if self.mode == 'dedup':
            seen = set()
            unique = []
            for clause in result:
                literals = frozenset(clause)
                if not literals or any(-lit in literals for lit in literals):
                    if not literals:
                        unique.append(clause)
                    continue
                if literals not in seen:
                    seen.add(literals)
                    unique.append(clause)
            result = unique
        self.counts['final'] = len(result)
        return result
