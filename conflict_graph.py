"""Sound graph lower bounds for completing a partial three-valued DFA.

A DONTCARE result imposes no condition on the completion. Only a common
suffix with opposite constrained results makes two states incompatible.
"""

from collections import deque


def incompatibility_graph(automaton):
    """Return (reachable state IDs, adjacency bitsets) for incompatible pairs.

    Reverse search in the synchronous product starts at ACCEPT/REJECT pairs.
    An undefined transition has no predecessor: it cannot witness a conflict.
    Only states reachable from an initial state are mapped by the SAT encoding.
    """
    states = sorted(automaton.get_reachable_states())
    size = automaton.num_states
    predecessors = [[[] for _ in range(size)] for _ in range(automaton.num_letters)]
    for source, transitions in enumerate(automaton.trans):
        for letter, target in transitions.items():
            predecessors[letter][target].append(source)

    seen = set()
    queue = deque()

    def add(left, right):
        if left > right:
            left, right = right, left
        if (left, right) not in seen:
            seen.add((left, right))
            queue.append((left, right))

    # The SAT encoding constrains both labels if a state is marked twice;
    # such an inconsistent input has no completion. For a valid input these
    # sets are disjoint (ACCEPT takes precedence in sdfa.run).
    for accept in automaton.final_states:
        for reject in automaton.reject_states:
            add(accept, reject)

    while queue:
        left, right = queue.popleft()
        for letter in range(automaton.num_letters):
            for p in predecessors[letter][left]:
                for q in predecessors[letter][right]:
                    add(p, q)

    position = {state: index for index, state in enumerate(states)}
    adjacency = [0] * len(states)
    for left, right in seen:
        if left != right and left in position and right in position:
            i, j = position[left], position[right]
            adjacency[i] |= 1 << j
            adjacency[j] |= 1 << i
    return states, adjacency


def _greedy_coloring(adjacency):
    n = len(adjacency)
    colors = [-1] * n
    for _ in range(n):
        vertex = max((v for v in range(n) if colors[v] < 0),
                     key=lambda v: (len({colors[u] for u in range(n)
                                         if adjacency[v] >> u & 1 and colors[u] >= 0}),
                                    adjacency[v].bit_count()))
        used = {colors[u] for u in range(n) if adjacency[vertex] >> u & 1 and colors[u] >= 0}
        colors[vertex] = next(c for c in range(n) if c not in used)
    return max(colors, default=-1) + 1


def _clique_bound(adjacency, budget):
    """A feasible clique at every point; a node budget only weakens the bound."""
    n = len(adjacency)
    best = 0
    steps = 0
    # Several greedy orders provide a cheap useful bound on dense graphs.
    starts = sorted(range(n), key=lambda v: adjacency[v].bit_count(), reverse=True)
    for start in starts[:64]:
        candidates = adjacency[start]
        count = 1
        while candidates:
            remaining = candidates
            v = -1
            score = -1
            while remaining:
                bit = remaining & -remaining
                remaining ^= bit
                candidate = bit.bit_length() - 1
                degree = (adjacency[candidate] & candidates).bit_count()
                if degree > score:
                    v, score = candidate, degree
            candidates &= adjacency[v]
            count += 1
        best = max(best, count)

    def visit(candidates, count):
        nonlocal best, steps
        if count > best:
            best = count
        if steps >= budget:
            return
        steps += 1
        while candidates and count + candidates.bit_count() > best and steps < budget:
            bit = candidates & -candidates
            v = bit.bit_length() - 1
            candidates ^= bit
            visit(candidates & adjacency[v], count + 1)

    visit((1 << n) - 1, 0)
    return best


def _not_k_colorable(adjacency, k, budget):
    """Return True only after an exhaustive failed k-color search."""
    n = len(adjacency)
    colors = [-1] * n
    steps = 0

    def search(used):
        nonlocal steps
        if steps >= budget:
            return None
        steps += 1
        uncolored = [v for v in range(n) if colors[v] == -1]
        if not uncolored:
            return True
        vertex = max(uncolored, key=lambda v: (
            len({colors[u] for u in range(n) if adjacency[v] >> u & 1 and colors[u] >= 0}),
            adjacency[v].bit_count()))
        forbidden = {colors[u] for u in range(n)
                     if adjacency[vertex] >> u & 1 and colors[u] >= 0}
        for color in range(min(used + 1, k)):
            if color in forbidden:
                continue
            colors[vertex] = color
            result = search(max(used, color + 1))
            colors[vertex] = -1
            if result is not False:
                return result
        return False

    return search(0) is False


def graph_lower_bound(automaton, exact_limit=40, search_budget=50000):
    """Return a certified chromatic lower bound, at least one for SAT search.

    For small graphs DSATUR proves consecutive k-coloring instances impossible,
    stopping at the chromatic number or the work budget. Larger graphs use a
    clique bound. A timeout never promotes an unproved coloring lower bound.
    """
    _, adjacency = incompatibility_graph(automaton)
    if not adjacency:
        return 1
    lower = max(1, _clique_bound(adjacency, search_budget))
    if len(adjacency) <= exact_limit:
        upper = _greedy_coloring(adjacency)
        while lower < upper:
            if not _not_k_colorable(adjacency, lower, search_budget):
                break
            lower += 1
    return lower
