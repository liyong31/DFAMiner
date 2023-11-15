# the standard way to import PySAT:
from pysat.formula import CNF
from pysat.solvers import Solver
from pysat.solvers import Glucose3,Cadical103,Cadical153,Gluecard4,Glucose42,Lingeling,MapleChrono

import itertools

from dfa import DFA,dict2dfa
from dfa.draw import write_dot


def is_even_loop(word, left, right):
    max_color = -1
    for j in range(left, right + 1, 1):
        max_color = max(max_color, word[j])
    
    if max_color % 2 == 1:
        return False
    else:
        return True

def get_membership(num_colors, word):
    positions = []
    for i in range(num_colors):
        positions.append((-1, -1))
    
    loop_mask = 0
    for i, color in enumerate(word):
        lft, rgt = positions[color]
        has_loop = False
        if lft == -1:
            positions[color] = (i, -1)
        elif rgt == -1:
            positions[color] = (lft, i)
            has_loop = True
        else:
            positions[color] = (rgt, i)
            has_loop = True
        
        lft, rgt = positions[color]
        is_even = is_even_loop(word, lft, rgt)
        #print("lft = " + str(lft) + ", rgt=" + str(rgt) + ", is_even=" + str(is_even) + ", has_loop=" + str(has_loop))
        if has_loop and is_even:
            loop_mask |= 2
        elif has_loop and not is_even:
            loop_mask |= 1
        #print("loop_mask=" + str(loop_mask))
        if loop_mask >= 3:
            return 0
    
    if loop_mask == 2:
        return 1
    elif loop_mask == 1:
        return -1
    else:
        return 0

def read_prefix_tree(name):
    f = open(name, "r")
    lines = f.readlines()
    if len(lines) < 2:
        print("file error: " + name)
        exit(-1)
    # first line is: #states, #letters
    line_brk = lines[0].split()
    num_states = int(line_brk[0])
    num_colors = int(line_brk[1])
    # print("#S =" + str(num_states) + " #COLORS =" + str(num_colors))
    g = [dict() for i in range(num_states)]

    line_brk = lines[1].split()
    init_state = int(line_brk[0])
    # print("init =" + str(init_state))

    for line in lines[2:]:
        line_brk = line.split()
        src_state = int(line_brk[0])
        letter = int(line_brk[1])
        dst_state = int(line_brk[2])
        # print("src = " + str(src_state) + " letter = " + str(letter) + " dst = " + str(dst_state))
        g[src_state][letter] = dst_state
    
    return (init_state, g)


# now we have trees, it is time to create numbers
# 0 <= i < j <= n-1
# 1. p_{j, i}: i is the parent of j in the BFS-tree
# 2. e_{a,i,j}: i goes to j over letter a
# 3. t_{i,j}: there is a transition from i to j
# 4. r_{a,i,j}: there is a transition from i to j over a and no smaller transition
def create_variables(n, alphabet, graph):
    
    num = 1
    # variables for DFA transitions
    prs = [(i, a, j) for i in range(n) for a in alphabet for j in range(n)]
    prs_m = prs
    edges = {element : (index + num) for index, element in enumerate(prs)}

    # variables for final states
    num += len(edges)
    sub_edges = { (i, -1, -1) : (num + i) for i in range(n)}

    edges.update(sub_edges)
    
    print("edge vars:\n" + str(edges))
    # variables for tree node and state
    num = len(edges) + 1
    prs = [(p, q) for p in range(len(graph)) for q in range(n)]
    nodes = { element : (index + num) for index, element in enumerate(prs)}
    print("node vars:\n" + str(nodes))
    
    # variables to encode BFS tree for DFA
    num += len(nodes)
    prs = [(p, q) for p in range(n) for q in range(n)]
    parents = { element : (index + num) for index, element in enumerate(prs)}
    print("para vars:\n" + str(parents))
    
    num += len(parents)
    t_aux = { element : (index + num) for index, element in enumerate(prs)}
    print("t vars:\n" + str(t_aux))
    
    num += len(t_aux)
    m_aux = { element : (index + num) for index, element in enumerate(prs_m)}
    print("m vars:\n" + str(m_aux))

    return (nodes, edges, parents, t_aux, m_aux)

def search_vertex(gr, init_state, w):
    # from init to the state
    curr_state = init_state
    curr_index = 0
    # print (str(w))
    while True:
        if curr_index >= len(w):
            return curr_state
        # otherwise, we move to next one
        # print("curr_state = " + str(curr_state))
        # print("curr_index = " + str(curr_index))
        # print("g[curr_state]: " + str(gr[curr_state]))
        # print("w = " + str(w))
        # print("prev index: " + str(curr_index))
        succ = gr[curr_state][w[curr_index]]
        curr_index += 1
        # print("after index: " + str(curr_index))
        curr_state = succ

def create_dfa_cnf(nodes, edges, init_state, n, alphabet, graph, pos, negs):
    
    clauses = []
    #A. deterministic transition formulas
    #A.1. not d_(p, a, q) or not d_(p, a, q')
    prs = [(p, a) for p in range(n) for a in alphabet]
    sub_clauses = []
    for p, a in prs:
        # must have one successor
        one_succ = [edges[p, a, q] for q in range(n)]
        sub_clauses = [one_succ] + sub_clauses
    clauses = sub_clauses + clauses
    
    #A.2. one edge over a letter
    diff_succs = [(q, qp) for q in range(n) for qp in range(n)]
    diff_succs = list(filter(lambda x: x[0] != x[1], diff_succs))
    print("diff_succ: " + str(diff_succs))
    sub_clauses = [ [0-edges[p,a,q], 0-edges[p,a,qp]] for (p, a) in prs for (q, qp) in diff_succs]
    clauses = sub_clauses + clauses
    
    #B. consistent with samples
    #B.1. first ensure that the initial state is empty word
    clauses += [[nodes[init_state, 0]]]
    clauses += [ [0-nodes[init_state, i]] for i in range(1, n)]
    #B.2. setup final states
    final_node = search_vertex(graph, init_state, pos[0])
    print("final node: " + str(final_node))
    sub_clauses = [[0-nodes[final_node, q], edges[q,-1,-1]] for q in range(n)]
    
    reject_node = search_vertex(graph, init_state, negs[0])
    print("reject node: " + str(reject_node))
    sub_clauses = [[0-nodes[reject_node, q], -edges[q,-1,-1]] for q in range(n)] + sub_clauses

    clauses = sub_clauses + clauses
    print("add clauses for samples")
    #B.3. consistent with samples
    prs = [ (nr, dr, letter) for (nr, dr) in enumerate(graph) for letter in alphabet]
    prs = list(filter(lambda x: (x[2] in x[1]), prs))
    #print (str(prs[0]))
    # check whether node has a child whose name is a
    # (nr, p) /\ edge(p, a, q) => (nr', q)
    sub_clauses = [[0-nodes[nr, p], 0-edges[p, letter, q], nodes[dr[letter], q]] 
                   for (nr, dr, letter) in prs for p in range(n) for q in range(n)]
    # (nr, p) /\ (nr', q) => edge(p, a, q) 
    #sub_clauses += [[0-nodes[nr, p], edges[p, letter, q], 0-nodes[dr[letter], q]] 
    #               for (nr, dr, letter) in prs for p in range(n) for q in range(n)]
    clauses = sub_clauses + clauses
    
    return clauses

# Symmetry breaking by enforcing a BFS-tree on the generated
# DFA, so the form is unique
# reference to "BFS-Based Symmetry Breaking Predicates
# for DFA Identification"
def create_BFStree_cnf(edges, parents, t_aux, m_aux, n, alphabet):
    
    clauses = []
    #C. node BFS-tree constraints
    #1. t_{i,j} <-> there is a transition from i to j
    prs = [(p, q) for p in range(n) for q in range(n)]
    #1.1 e(p,a,q) => t(p,q)
    sub_cluases = [ [0 - edges[p, a, q], t_aux[p, q]] for a in alphabet for (p, q) in prs]
    #1.2 t(p, q) => some e(p, a, q)
    for p, q in prs:
        edge_rel = [edges[p, a, q] for a in alphabet]
        edge_rel = [0- t_aux[p, q]] + edge_rel
        sub_cluases = [edge_rel] + sub_cluases
    
    clauses = sub_cluases + clauses
    
    #2. BFS tree order
    #2.1 p_{j, i} i is the parent of j
    sub_cluases = []
    for j in range(1, n):
        # only one p_{j, 0}, p_{j,1} ... only one parent
        
        one_parent = [ parents[j, i] for i in range(j)]
        sub_cluases = [one_parent] + sub_cluases 
        
        # p_{j,i} => t_{i,j}
        exist_edges = [ [0- parents[j, i], t_aux[i, j]] for i in range(j)]
        sub_cluases = exist_edges + sub_cluases
        
        # t_{i,j} /\ !t_{i-1, j} /\ !t_{i-2,j} /\ ... /\ !t_{0,j} => p_{j,i}
        for i in range(j):
            no_smaller_pred = [ t_aux[k, j] for k in range(i)]
            no_smaller_pred = [0-t_aux[i, j], parents[j,i] ] + no_smaller_pred
            sub_cluases = [no_smaller_pred] + sub_cluases
            
    clauses = sub_cluases + clauses
        
    ij_pairs = [ (i, j) for j in range(n) for i in range(n)]
    ij_pairs = list(filter(lambda x: x[0] < x[1], ij_pairs))
    
    sub_cluases = []
    for i, j in ij_pairs:
        # only k < i < j, p_{j,i} => ! t_{k,j}
        # if i is parent of j in the BFS-tree, then no edge from k to j
        # otherwise k will traverse j first 
        k_vals = list(range(i))
        no_smaller_edges = [ [0-parents[j,i], 0-t_aux[k, j]] for k in k_vals]
        sub_cluases = no_smaller_edges + sub_cluases
    
    clauses = sub_cluases + clauses
    
    #3. relation to edges
    # m_{i, a, j} => e_{i, a, j}
    edge_rel = [ [0-m_aux[i, a, j], edges[i, a, j]] for a in alphabet for (i, j) in ij_pairs]
    
    kh_pairs = [ (k, h) for h in alphabet for k in alphabet]
    kh_pairs = list(filter(lambda x: x[0] < x[1], kh_pairs))
    # larger h > k, m_{i, h, j} => ! e_{i, k, j}
    # if there is a larger letter over i -> j in the BFS-tree,
    # then there must be no edge from i to j over a smaller letter
    edge_rel = [ [0-m_aux[i, h, j], 0-edges[i, k, j]] 
                for (k, h) in kh_pairs for (i, j) in ij_pairs] + edge_rel
    clauses = edge_rel + clauses
    
    # for every (i,j), e_{i,h,j} /\ ! e_{i,h-1,j} /\ .../\ !e_{i,0,j} => m_{i,h,j}
    # that is, if h is the smallest letter from i to j, then it is in BFS tree
    sub_cluases = []
    prs = [ (i, j, a) for (i,j) in ij_pairs for a in alphabet]
    for i, j, a in prs:
        edge_rel = [edges[i, h, j] for h in range(a)] # smaller letters
        edge_rel = [0-edges[i, a, j], m_aux[i, a, j]] + edge_rel
        sub_cluases = [edge_rel] + sub_cluases
    
    clauses = sub_cluases + clauses
    
    #4. BFS tree parent-child relation
    ijk_pairs = [ (k, i, j) for k in range(n-1) for i in range(n-1) for j in range(n-1)]
    ijk_pairs = list(filter(lambda x: x[0] < x[1] and x[1] < x[2], ijk_pairs))
    # p_{j, i} => !p_{j+1, k}, it means that i is parent of j, then k is not possible to be the parent of j + 1
    # since k is even smaller than i 
    edge_rel = [ [0- parents[j,i], 0-parents[j+1, k]] for (k, i, j) in ijk_pairs]

    ij_pairs = [ (i, j) for j in range(n-1) for i in range(n-1)]
    ij_pairs = list(filter(lambda x: x[0] < x[1], ij_pairs))
    # p_{j,i} /\ p_{j+1, i} /\ m_{i,h,j} => !m_{i,k,j+1}
    # if i is parent of both j and j + 1, and in the BFS-tree, we have i ->j over h,
    # then there is no smaller letter k from i to j+1?
    edge_rel = [[ 0-parents[j,i], 0-parents[j+1, i], 0-m_aux[i, h, j], 0-m_aux[i, k, j+1]]
                for (i, j) in ij_pairs for (k, h) in kh_pairs] + edge_rel
    
    clauses = edge_rel + clauses
    
    return clauses
    
def create_cnf(nodes, edges, parents, t_aux, m_aux, init_state, n, alphabet, graph, pos, negs):
    clauses = create_dfa_cnf(nodes, edges, init_state, n, alphabet, graph, pos, negs)
    sub_clauses = create_BFStree_cnf(edges, parents, t_aux, m_aux, n, alphabet)
    clauses = sub_clauses + clauses
    for cls in clauses:
        print(cls)
    return clauses

def construct_dfa_from_model(model, edges, n):
    # print("type(model) = " + str(type(model)))
    dfa_dict = dict()
    for p in range(n):
        # print("State " + str(p))
        is_final = False
        if model[edges[p, -1, -1] - 1] > 0:
            # print(" final")
            is_final = True
        trans = dict()
        for a, letter in enumerate(alphabet):
            # print("a =" +str(a) + ", letter=" + str(letter))
            for q in range(n):
                if model[edges[p, a, q] -1 ] > 0:
                    # print("var = " + str(e1[p, a, q]))
                    # print("letter " + str(letter) + " -> " + str(q))
                    trans[letter] = q
        dfa_dict[p] = (is_final, trans)

    dfa = dict2dfa(dfa_dict, start=0)
    write_dot(dfa, "./dfa1.dot")
    return dfa
    
def solve(n, alphabet, init_state, graph, pos, negs):
    
    nodes, edges, parents, t_aux, m_aux = create_variables(n, alphabet, graph)
    # solvers, Glucose3(), Cadical103(), Cadical153(), Gluecard4(), Glucose42()
    g = Cadical153() #Lingeling() #Glucose3()

    clauses = create_cnf(nodes, edges, parents, t_aux, m_aux, init_state, n, alphabet, graph, pos, negs)
    print("#Vars: " + str(len(nodes) + len(edges) + len(parents) + len(t_aux) + len(m_aux)))
    print("#Clauses: " + str(len(clauses)))
    print("#N: " + str(n))
    print("#Pos: " + str(len(pos)))
    print("#Neg: " + str(len(negs)))
    for cls in clauses:
        g.add_clause(cls)
    
    is_sat = g.solve()
    print(is_sat)

    if is_sat:
        print(g.get_model())

        model= g.get_model()
        # now print out transition relation
        # print("type(model) = " + str(type(model)))
        dfa = construct_dfa_from_model(model, edges, n)
        return (True, dfa)
    else:
        return (False, None)    
                
# need to check whether the DFA accepts odd word
# or reject some even word
# a word is even iff all the cylcles in it are even
# similarly a word is odd iff all the cylces are odd
# so there are some word that is neither even nor odd
def verify_conjecture_dfa(num_colors, dfa, pos, negs):
    # we verify by enumerating all possible words
    
    for p in pos:
        print("word: = " + str(p))
        print("label: = " + str( dfa.label(p)))
        if not dfa.label(p):
            return (False, p)
    for p in negs:
        print("word: = " + str(p))
        print("label: = " + str( dfa.label(p)))
        if dfa.label(p):
            return (False, p)
    
    return (True, [])

def repeat_k(k, num):
    res = []
    for i in range(num):
        res = [k] + res
    return res



def get_word(line):
    line_brk = line.split()
    mq = int(line_brk[0])
    num = int(line_brk[1])
    w = [int(i) for i in line_brk[2:(2+ num)]]
    return [mq] + w

def read_samples(name):
    f = open(name, "r")
    lines = f.readlines()
    if len(lines) < 2:
        print("file error: " + name)
        exit(-1)
    # first line is: #samples, #letters
    line_brk = lines[0].split()
    num_samples = int(line_brk[0])
    num_colors = int(line_brk[1])
    # print("#S =" + str(num_states) + " #COLORS =" + str(num_colors))
    g = [get_word(line) for line in lines[1:]]
    pos = filter(lambda x: x[0] == 1, g)
    pos = [ w[1:] for w in pos]
    negs = filter(lambda x: x[0] == 0, g)
    negs = [ w[1:] for w in negs]
    dont = filter(lambda x: x[0] == -1, g)
    dont = [w[1:] for w in dont]
    return (pos, negs, dont)
     
num_colors = 6
alphabet = list(range(num_colors))
num_letters = 7
prefix = "data" + str(num_colors) + "-" + str(num_letters) + "-all"
pos, negs, dont = read_samples(prefix + ".txt")
print("Positive: ")
for p in pos:
    print(p)

print("Negative: ")
for p in negs:
    print(p)
    
print("Dontcares: ")
# for p in dont:
#     print(p)


n = 20
file_name = prefix + "-dfa.txt"
init_state, graph = read_prefix_tree(file_name)

while True:
    print("Iteration " + str(n))
    print("graph size: " + str(len(graph)))
    res, dfa = solve(n, alphabet, init_state, graph, pos, negs)
    if res:
        # it has been successful
        print("Verifying DFA conjecture...")
        eq_res, cex = verify_conjecture_dfa(len(alphabet), dfa, pos, negs)
        dfa_file = "./dfa" + str(num_colors) + "-" + str(num_letters) + "-nn.dot"
        print("Output to " + dfa_file)
        write_dot(dfa, dfa_file)
        if eq_res:
            print("Status: Correct DFA")
            break
        else:
            print("Status: Wrong DFA")
            mq_res = get_membership(len(alphabet), cex)
            if mq_res == 1:
                pos = [cex] + pos
            elif mq_res == -1:
                negs = [cex] + negs
            else:
                print("error cex: " + str(cex))
            exit(-1)
    else:
        n += 1