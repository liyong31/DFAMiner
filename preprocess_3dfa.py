"""Sound, low-cost preprocessing and input-guided SAT symmetry breaking.

The input 3DFA is a deterministic partial automaton with two kinds of
mandatory terminal observations: accept and reject.  A state can be dropped
when it is not on any rooted path to a labelled state, since it cannot
contribute any classified sample.  All initial states are preserved, even
when no label is reachable from one of them.

The SAT target has a separate symmetry: target states can be renamed,
provided state 0 remains the initial state.  For any ordered sequence of
root-reachable input states starting at an initial state, the first
appearance of each target state ID may be assumed to follow the order
0, 1, 2, ... . This is the restricted-growth-string (RGS) symmetry
breaker. The input 3DFA itself need not have any symmetry.

The RGS encoding is valid for arbitrary target DFA sizes, including
sizes above the optimum. It must not be combined with the existing
*target-BFS canonical numbering*: those two permutations need not have
a common representative.

No expensive pairwise incompatibility relation is computed.
"""
from collections import deque
from time import perf_counter


def trim_observation_irrelevant(machine):
    """Return root-and-observation-preserving induced sub-3DFA and metadata.

    Complexity is O(|Q|+|delta|). For every initial state and word, the
    label observed in the returned automaton is identical to the original.
    Removing an edge to a state that cannot reach a label is safe: from
    there every continuation is 'don't care' anyway.

    The reference sdfa.reduce() may remove initial states entirely when
    their components contain no labels, so do not use it here.
    """
    start = perf_counter()
    if not machine.init_states:
        return machine, dict(original_states=machine.num_states,
                             kept_states=machine.num_states,
                             removed_states=0,
                             elapsed_seconds=perf_counter()-start)
    reachable = set(machine.get_reachable_states())
    predecessor = [[] for _ in range(machine.num_states)]
    for p, outgoing in enumerate(machine.trans):
        for q in outgoing.values():
            predecessor[q].append(p)
    labelled = set(machine.final_states) | set(machine.reject_states)
    co_reachable = set(labelled)
    queue = deque(labelled)
    while queue:
        state = queue.popleft()
        for p in predecessor[state]:
            if p not in co_reachable:
                co_reachable.add(p)
                queue.append(p)

    relevant = (reachable & co_reachable) | set(machine.init_states)
    if len(relevant) == machine.num_states:
        return machine, dict(original_states=machine.num_states,
                             kept_states=machine.num_states,
                             removed_states=0,
                             elapsed_seconds=perf_counter()-start)

    # Preserve deterministic state numbering and original root order. No
    # input state is merged here, so this never introduces new paths.
    ordered = sorted(relevant)
    old_to_new = {p:i for i,p in enumerate(ordered)}
    result = type(machine)()
    result.set_num_states(len(ordered))
    result.set_num_letters(machine.num_letters)
    for p in ordered:
        q = old_to_new[p]
        if p in machine.init_states:
            result.add_initial_state(q)
        if p in machine.final_states:
            result.add_final_state(q)
        if p in machine.reject_states:
            result.add_reject_state(q)
        for a,dst in machine.trans[p].items():
            if dst in old_to_new:
                result.add_transition(q,a,old_to_new[dst])
    return result, dict(original_states=machine.num_states,
                        kept_states=result.num_states,
                        removed_states=machine.num_states-result.num_states,
                        removed_edges=(machine.get_num_trans() -
                                       result.get_num_trans()),
                        elapsed_seconds=perf_counter()-start)


def ordered_reachable_states(machine, policy='bfs'):
    """Give rooted states a deterministic order with an initial root first."""
    if not machine.init_states:
        return []
    reachable = machine.get_reachable_states()
    roots = sorted(machine.init_states & reachable)
    if not roots:
        return []
    if policy == 'bfs':
        queue = deque(roots)
        visited = set(roots)
        result = list(roots)
        while queue:
            p = queue.popleft()
            for a in sorted(machine.trans[p]):
                q = machine.trans[p][a]
                if q in reachable and q not in visited:
                    visited.add(q)
                    result.append(q)
                    queue.append(q)
        return result
    if policy not in ('fanin','label'):
        raise ValueError('unknown source-state ordering: '+policy)
    in_degree = [0]*machine.num_states
    for outgoing in machine.trans:
        for q in outgoing.values():
            in_degree[q] += 1
    labels = machine.final_states | machine.reject_states
    others = reachable - set(roots)
    if policy == 'fanin':
        priority = lambda p:(-in_degree[p], -(p in labels), p)
    else:
        priority = lambda p:(-(p in labels), -in_degree[p], p)
    return roots+sorted(others,key=priority)


def rgs_symmetry_clauses(nodes, machine, k, policy='bfs', prefix_size=32):
    """Add an input-guided restricted-growth target-state numbering.

    Input mapping x[p,i] is uniquely determined on root-reachable states
    by the deterministic target transitions and initial-state equations.
    Relabel target states by order of *first occurrence in the selected
    prefix* (root maps to 0). Hence any solution has an isomorphic
    representative satisfying these clauses.

    For the t-th observed input state p:
       x[p,c] -> (OR_{s<t} x[p_s,c-1]), for c>=2.
    For c>t, the implication is tightened to -x[p,c], as c cannot
    appear this early in a restricted growth string.
    There is no need for c=1: initial root p_0 already maps to 0.

    This is distinct from the full target-BFS constraint and must be
    tested separately; adding both could destroy completeness.
    """
    if k < 1 or prefix_size < 1:
        return []
    order = ordered_reachable_states(machine,policy)[:prefix_size]
    if len(order) < 2:
        return []
    result = []
    for t in range(1,len(order)):
        p = order[t]
        for c in range(2,k):
            if c > t:
                result.append([-nodes[p,c]])
            else:
                result.append([-nodes[p,c]] +
                              [nodes[order[s],c-1] for s in range(t)])
    return result
