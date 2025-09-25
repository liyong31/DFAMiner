# the standard way to import PySAT:
from pysat.formula import CNF
from pysat.solvers import Solver

import sys
import time

import sdfa as SDFA


class safe_miner:
    
    def __init__(self) -> None:
        self.num_vars = 0
    # now we have trees, it is time to create numbers
    # 0 <= i < j <= n-1
    # 1. p_{j, i}: i is the parent of j in the BFS-tree
    # 2. e_{a,i,j}: i goes to j over letter a
    # 3. t_{i,j}: there is a transition from i to j
    # 4. r_{a,i,j}: there is a transition from i to j over a and no smaller transition


    def create_variables(self, n, alphabet, pos_num, neg_num, neg_accep_states, reach):

        num = 1
        # variables for DFA transitions
        prs = [(i, a, j) for i in range(n) for a in alphabet for j in range(n)]
        prs_m = prs
        edges = {element: (index + num) for index, element in enumerate(prs)}

        # print("edge vars:\n" + str(edges))
        # variables for positive tree node and state
        num += len(edges)
        prs = [(p, q) for p in range(pos_num) for q in range(n)]
        pos_nodes = {element: (index + num) for index, element in enumerate(prs)}
        # print("posnode vars:\n" + str(pos_nodes))
        # variables for positive tree node and state
        num += len(pos_nodes)
        prs = [(p, q) for p in range(neg_num) for q in range(n)]
        neg_nodes = {element: (index + num) for index, element in enumerate(prs)}
        # print("negnode vars:\n" + str(neg_nodes))

        # variables to encode BFS tree for DFA
        num += len(neg_nodes)
        prs = [(p, q) for p in range(n) for q in range(n)]
        parents = {element: (index + num) for index, element in enumerate(prs)}
        # print("para vars:\n" + str(parents))

        num += len(parents)
        t_aux = {element: (index + num) for index, element in enumerate(prs)}
        # print("t vars:\n" + str(t_aux))

        num += len(t_aux)
        m_aux = {element: (index + num) for index, element in enumerate(prs_m)}
        # print("m vars:\n" + str(m_aux))
        
        num += len(m_aux)
        # for rejecting negative samples
        # here we assume state n-1 is the sink state, we only care about accepting states here
        neg_reach = None
        if not reach:
            neg_reach = [(p1, q1, p2, q2) for p1 in neg_accep_states for p2 in neg_accep_states
                        for q1 in range(n-1) for q2 in range(n-1)]
        else:
            # novel reachability relation, only use target pairs
            neg_reach = [ (q1, q2) for q1 in range(n-1) for q2 in range(n-1)]
        neg_aux = {element : (index + num) for index, element in enumerate(neg_reach)}
        # print("reach vars:\n", neg_aux)
        print("#Number of vars: ", num + len(neg_aux))
        self.num_vars = num + len(neg_aux)

        return (pos_nodes, neg_nodes, edges, parents, t_aux, m_aux, neg_aux)
    
    
    def make_deterministic_complete(self, edges, n, alphabet):
        clauses = []
        
        # A. deterministic transition formulas
        # A.1. not d_(p, a, q) or not d_(p, a, q')
        prs = [(p, a) for p in range(n) for a in alphabet]
        sub_clauses = []
        for p, a in prs:
            # must have one successor
            one_succ = [edges[p, a, q] for q in range(n)]
            sub_clauses = [one_succ] + sub_clauses
        clauses = sub_clauses + clauses

        # A.2. one edge over a letter
        diff_succs = [(q, qp) for q in range(n) for qp in range(n)]
        diff_succs = list(filter(lambda x: x[0] != x[1], diff_succs))
        # print("diff_succ: " + str(diff_succs))
        sub_clauses = [[0-edges[p, a, q], 0-edges[p, a, qp]]
                    for (p, a) in prs for (q, qp) in diff_succs]
        clauses = sub_clauses + clauses
        
        # print("det-complete:", clauses)
        
        return clauses
    
    def make_safe(self, edges, n, alphabet):
        # n-1 is the sink state
        return [[edges[n-1, a, n-1]] for a in alphabet]
    
    def make_transiton_consistency(self, posnodes, negnodes, edges, n, alphabet, pos_dba, neg_dba):
    
        # B.3. consistent with samples
        pos_prs = [(nr, dr, letter) for (nr, dr) in enumerate(pos_dba.trans)
            for letter in alphabet]
        # if the input is complete, then we will not have this issue
        pos_prs = list(filter(lambda x: (x[2] in x[1]), pos_prs))
        # print(pos_prs)

        neg_prs = [(nr, dr, letter) for (nr, dr) in enumerate(neg_dba.trans)
            for letter in alphabet]
        neg_prs = list(filter(lambda x: (x[2] in x[1]), neg_prs))
        # print(neg_prs)

        # print (str(prs[0]))
        # check whether node has a child whose name is a
        # (nr, p) /\ edge(p, a, q) => (nr', q)
        clauses = [[0-posnodes[nr, p], 0-edges[p, letter, q], posnodes[dr[letter], q]]
                    for (nr, dr, letter) in pos_prs for p in range(n) for q in range(n)]

        sub_clauses = [[0-negnodes[nr, p], 0-edges[p, letter, q], negnodes[dr[letter], q]]
                    for (nr, dr, letter) in neg_prs for p in range(n) for q in range(n)]
        clauses += sub_clauses
        
        # initial states
        sub_clauses = [[posnodes[nr, 0]] for nr in pos_dba.init_states] + [[negnodes[nr, 0]] for nr in neg_dba.init_states]
        clauses += sub_clauses
        
        # print("cons:", clauses)
        
        return clauses
    
    def accept_positive_samples(self, nodes, n, pos_dba):
        sink_reject_states = pos_dba.reject_states
        accept_states = { s for s in range(pos_dba.num_states) if not (s in sink_reject_states) }
        clauses = [ [ -nodes[s, n-1]] for s in accept_states]
        return clauses
    
    def reject_negative_samples(self, nodes, edges, neg_aux, reach, n, alphabet, neg_dba):
        
        clauses = []
        accept_states = { s for s in range(neg_dba.num_states) if not (s in neg_dba.reject_states) }
        tgt_states_triples = [(q1, q2, q3) for q1 in range(n-1) for q2 in range(n-1) for q3 in range(n-1)]
        ref_states_pairs = [(p1, p2) for p1 in accept_states for p2 in accept_states]
        tgt_state_pairs = [(q1, q2) for q1 in range(n-1) for q2 in range(n-1)]

        if not reach:
            # q1 is any state, q2 is not sink, q1 /= q3
            ref_states_pairs = [(p1, p2) for p1 in accept_states for p2 in accept_states]
            
            # reachability relations
            # (p1, q1, p2, q2) /\ e_{q2, a, q3} => (p1, q1, T(p2, a), q3)
            # T(p2, a) not sink, T(p2, a) /= p1 does not close a loop 
            sub_clauses = [[-neg_aux[p1, q1, p2, q2], -edges[q2, a, q3], neg_aux[p1, q1, neg_dba.trans[p2][a], q3]]
                    for (q1, q2, q3) in tgt_states_triples 
                    for (p1, p2) in ref_states_pairs
                    for a in alphabet if not (neg_dba.trans[p2][a] in neg_dba.reject_states) and (neg_dba.trans[p2][a] != p1 or q1 != q3) ]
            clauses += sub_clauses
            
            # q1 is a safe state
            # tgt_state_pairs = [(q1, q2) for q1 in range(n-1) for q2 in range(n-1)]

            # not (p1, q1, p2, q2) or not e_(q_1, a, q2)
            # contrary to this: when both close a loop, then they both are in the accepting region
            clauses += [[-neg_aux[p1, q1, p2, q2], -edges[q2, a, q1]]
                    for (q1, q2) in tgt_state_pairs 
                    for (p1, p2) in ref_states_pairs
                    for a in alphabet if (neg_dba.trans[p2][a] == p1) ]
            # reachable states
            pair = [ (p, q) for p in accept_states for q in range(n-1)]
            # d_(p, q) => x_(p, q, p, q)
            clauses += [ [-nodes[p, q], neg_aux[p, q, p, q]] for (p, q) in pair]
            # not d_{p, q} => not x_(p, q, p1, q1) for all successors
            clauses += [ [nodes[p, q], -neg_aux[p, q, p1, q1]] for (p, q) in pair for (p1, q1) in pair]

            # acceptance conditions: (p, q, p, q') /\ p not sink => q' the sink
            # equivalent to: q' is not the sink => not (p, q, p, q') or p is sink 
        else:
            print(neg_aux)
            dba_aux = {pair : (index + self.num_vars) for index, pair in enumerate(ref_states_pairs)}
            self.num_vars += len(dba_aux)
            print("Num of vars:", self.num_vars, len(dba_aux))
            print(dba_aux)
            # now we need to add transition relation for the pairs (p, q) where p can reach q without visiting
            # sink state also, the word can run in reference automaton
            # d_{p2, q_2} /\ (r_{q1, q_2} /\ [\/_{a in alphabet} e_{q2, a, q_3})] => r_{q_1, q3}
            sub_clauses = [[-nodes[p1, q1], -neg_aux[q1, q2], -dba_aux[p1, p2], -edges[q2, a, q3], neg_aux[q1, q3]]
                           for (q1, q2, q3) in tgt_states_triples
                           for (p1, p2) in ref_states_pairs
                           for a in alphabet if (neg_dba.trans[p2][a] in accept_states) and (neg_dba.trans[p2][a] != p1 or q1 != q3) ]
            clauses += sub_clauses

            sub_clauses = [[-nodes[p1, q1], -neg_aux[q1, q2], -dba_aux[p1, p2], -edges[q2, a, q3], dba_aux[p1, neg_dba.trans[p2][a]]]
                           for (q1, q2, q3) in tgt_states_triples
                           for (p1, p2) in ref_states_pairs
                           for a in alphabet if (neg_dba.trans[p2][a] in accept_states) and (neg_dba.trans[p2][a] != p1 or q1 != q3) ]
            clauses += sub_clauses
            # clauses += [[-nodes[q1, q2], -neg_aux[p2, q2], -edges[q2, a, q3], -neg_aux[q1, q3]]
            #         for (q1, q2, q3) in tgt_states_triples 
            #         for p2 in accept_states
            #         for a in alphabet if (neg_dba.trans[p2][a] in neg_dba.reject_states)]

            # Close a loop and have overlap: d_{p1, q1} /\ d_{p2, q2} /\ p1 = T(p2, a) /\ e_{q2, a, q1} /\ r_{q1, q2} 
            # contrary to this: when both close a loop, then they both are in the accepting region
            clauses += [[-nodes[p1, q1], -nodes[p2, q2], -neg_aux[q1, q2], -dba_aux[p1, p2], -edges[q2, a, q1]]
                    for (q1, q2) in tgt_state_pairs 
                    for (p1, p2) in ref_states_pairs
                    for a in alphabet if (neg_dba.trans[p2][a] == p1) ]
            # reachable states
            # pair = [ (q, q) for q in range(n-1)]
            # all states can reach itself
            clauses += [[neg_aux[0, 0]]]
            clauses += [[dba_aux[p, p]] for p in neg_dba.init_states]
            # pair = [ (p, q) for p in accept_states for q in range(n-1)]
            # clauses += [ [-nodes[p, q], neg_aux[q, q]] for (p, q) in pair]
            # clauses += [ [-nodes[p, q], dba_aux[p, p]] for (p, q) in pair]

        return clauses

    def create_dfa_cnf(self, nodes, edges, input_sdfa, n, alphabet):

        clauses = []
        

        # B. consistent with samples
        # B.1. first ensure that initial states are empty word
        clauses += [[nodes[init, 0]] for init in input_sdfa.init_states]
        clauses += [[0-nodes[init, i]] for i in range(1, n) for init in input_sdfa.init_states]
        # B.2. setup final states
        # final_node = search_vertex(graph, init_state, pos[0])
        # print("final node: " + str(final_node))
        sub_clauses = [[0-nodes[final_node, q], edges[q, -1, -1]]
                    for q in range(n) for final_node in input_sdfa.final_states]

        # reject_node = search_vertex(graph, init_state, negs[0])
        # print("reject node: " + str(reject_node))
        sub_clauses = [[0-nodes[reject_node, q], -edges[q, -1, -1]]
                    for q in range(n) for reject_node in input_sdfa.reject_states] + sub_clauses

        clauses = sub_clauses + clauses
        # print("add clauses for samples")
        # B.3. consistent with samples
        prs = [(nr, dr, letter) for (nr, dr) in enumerate(input_sdfa.trans)
            for letter in alphabet]
        prs = list(filter(lambda x: (x[2] in x[1]), prs))
        # print (str(prs[0]))
        # check whether node has a child whose name is a
        # (nr, p) /\ edge(p, a, q) => (nr', q)
        sub_clauses = [[0-nodes[nr, p], 0-edges[p, letter, q], nodes[dr[letter], q]]
                    for (nr, dr, letter) in prs for p in range(n) for q in range(n)]
        # (nr, p) /\ (nr', q) => edge(p, a, q)
        # sub_clauses += [[0-nodes[nr, p], edges[p, letter, q], 0-nodes[dr[letter], q]]
        #               for (nr, dr, letter) in prs for p in range(n) for q in range(n)]
        clauses = sub_clauses + clauses

        #EXT. if they are two separate DFAs, we can just add
        # (s, p) and (t, p) cannot hold at the same time for final s and reject t
        # if they can reach the same states, then they must be not final and not reject
        # if len(init_state) == 2:
        #     max_init = max(init_state)
        #     max_state = max([ dr[letter] for (_, dr, letter) in prs])
        #     # s is final and t is reject, cannot reach the same state in DFA
        sub_clauses = [ [ -nodes[s, p], -nodes[t, p]] 
            for s in input_sdfa.final_states # not accept
            for t in input_sdfa.reject_states
            for p in range(0, n)] # the first posDFA

        clauses = sub_clauses + clauses
            

        return clauses

    # Symmetry breaking by enforcing a BFS-tree on the generated
    # DFA, so the form is unique
    # reference to "BFS-Based Symmetry Breaking Predicates
    # for DFA Identification"


    def create_BFStree_cnf(self, edges, parents, t_aux, m_aux, n, alphabet, safety):
        print("adding constraints for BFS tree...")
        clauses = []
        # C. node BFS-tree constraints
        # 1. t_{i,j} <-> there is a transition from i to j
        prs = [(p, q) for p in range(n) for q in range(n)]
        # 1.1 e(p,a,q) => t(p,q)
        sub_cluases = [[0 - edges[p, a, q], t_aux[p, q]]
                    for a in alphabet for (p, q) in prs]
        # 1.2 t(p, q) => some e(p, a, q)
        for p, q in prs:
            edge_rel = [edges[p, a, q] for a in alphabet]
            edge_rel = [0 - t_aux[p, q]] + edge_rel
            sub_cluases = [edge_rel] + sub_cluases

        clauses = sub_cluases + clauses

        # 2. BFS tree order
        # 2.1 p_{j, i} i is the parent of j
        sub_cluases = []
        for j in range(1, n):
            # only one p_{j, 0}, p_{j,1} ... only one parent

            one_parent = [parents[j, i] for i in range(j)]
            sub_cluases = [one_parent] + sub_cluases

            # p_{j,i} => t_{i,j}
            exist_edges = [[0 - parents[j, i], t_aux[i, j]] for i in range(j)]
            sub_cluases = exist_edges + sub_cluases

            # t_{i,j} /\ !t_{i-1, j} /\ !t_{i-2,j} /\ ... /\ !t_{0,j} => p_{j,i}
            for i in range(j):
                no_smaller_pred = [t_aux[k, j] for k in range(i)]
                no_smaller_pred = [0-t_aux[i, j], parents[j, i]] + no_smaller_pred
                sub_cluases = [no_smaller_pred] + sub_cluases

        clauses = sub_cluases + clauses

        ij_pairs = [(i, j) for j in range(n) for i in range(n)]
        ij_pairs = list(filter(lambda x: x[0] < x[1], ij_pairs))

        sub_cluases = []
        for i, j in ij_pairs:
            # only k < i < j, p_{j,i} => ! t_{k,j}
            # if i is parent of j in the BFS-tree, then no edge from k to j
            # otherwise k will traverse j first
            k_vals = list(range(i))
            no_smaller_edges = [[0-parents[j, i], 0-t_aux[k, j]] for k in k_vals]
            sub_cluases = no_smaller_edges + sub_cluases

        clauses = sub_cluases + clauses

        # 3. relation to edges
        # m_{i, a, j} => e_{i, a, j}
        edge_rel = [[0-m_aux[i, a, j], edges[i, a, j]]
                    for a in alphabet for (i, j) in ij_pairs]

        kh_pairs = [(k, h) for h in alphabet for k in alphabet]
        kh_pairs = list(filter(lambda x: x[0] < x[1], kh_pairs))
        # larger h > k, m_{i, h, j} => ! e_{i, k, j}
        # if there is a larger letter over i -> j in the BFS-tree,
        # then there must be no edge from i to j over a smaller letter
        edge_rel = [[0-m_aux[i, h, j], 0-edges[i, k, j]]
                    for (k, h) in kh_pairs for (i, j) in ij_pairs] + edge_rel
        clauses = edge_rel + clauses

        # for every (i,j), e_{i,h,j} /\ ! e_{i,h-1,j} /\ .../\ !e_{i,0,j} => m_{i,h,j}
        # that is, if h is the smallest letter from i to j, then it is in BFS tree
        sub_cluases = []
        prs = [(i, j, a) for (i, j) in ij_pairs for a in alphabet]
        for i, j, a in prs:
            edge_rel = [edges[i, h, j] for h in range(a)]  # smaller letters
            edge_rel = [0-edges[i, a, j], m_aux[i, a, j]] + edge_rel
            sub_cluases = [edge_rel] + sub_cluases

        clauses = sub_cluases + clauses

        max_state = -1
        if safety: 
            max_state = n - 2 
        else: 
            max_state = n - 1
        # 4. BFS tree parent-child relation
        ijk_pairs = [(k, i, j) for k in range(n-1)
                    for i in range(n-1) for j in range(max_state)]
        ijk_pairs = list(filter(lambda x: x[0] < x[1] and x[1] < x[2], ijk_pairs))
        # p_{j, i} => !p_{j+1, k}, it means that i is parent of j, then k is not possible to be the parent of j + 1
        # since k is even smaller than i
        edge_rel = [[0 - parents[j, i], 0-parents[j+1, k]]
                    for (k, i, j) in ijk_pairs]

        ij_pairs = [(i, j) for j in range(max_state) for i in range(n-1)]
        ij_pairs = list(filter(lambda x: x[0] < x[1], ij_pairs))
        # (6)
        # p_{j,i} /\ p_{j+1, i} /\ m_{i,h,j} => !m_{i,k,j+1}
        # if i is parent of both j and j + 1, and in the BFS-tree, we have i ->j over h,
        # then there is no smaller letter k from i to j+1?
        edge_rel = [[0-parents[j, i], 0-parents[j+1, i], 0-m_aux[i, h, j], 0-m_aux[i, k, j+1]]
                    for (i, j) in ij_pairs for (k, h) in kh_pairs] + edge_rel

        clauses = edge_rel + clauses

        return clauses
    
    def create_symmetry_break(self, edges, n, alphabet):
        pairs = [(p, q) for p in range(n-1) for q in range(n) for q in range(n) if p < q]
        return [[-edges[p, a, q]] for (p, q) in pairs for a in alphabet if a > (p) * n + q + 3]

    # here we define the DSA constraints
    # DSA with n-1 safe states and 1 rejecting sink
    # 1. sink states only have odd incoming transitions
    # 2. All states but sink loop on even colours
    # 3. initial state has loops on all even colours
    # 4. if maximal letter is even, all states except sink goes back to 0
    def create_DSA_cnf(self, edges, n, alphabet):

        print("adding constraints for solving parity games...")
        clauses = []

        #1. this should be true
        odd_colors = [ num for num in alphabet if num % 2 != 0]
        even_colors = [ num for num in alphabet if num % 2 == 0]
        
        # 1. sink state only have odd incoming transitions
        clauses += [[-edges[s, el, n-1]] for el in even_colors for s in range(n-1)]
        # 2. will not loop on odd letters 
        clauses += [[-edges[s, ol, s]] for ol in odd_colors for s in range(n-1)]
        # 3. the initial state loop on all even letters 
        clauses += [[edges[0, el, 0]] for el in even_colors]
        # 4. if maximal letter is even, all states except sink goes back to 0
        max_letter = max (alphabet)
        if max_letter % 2 == 0:
                clauses += [ [edges[s, max_letter, 0]] for s in range(0, n-1)]
        # 5. initial state 0 must go to other state (not sink) over odd letters        
        clauses += [[-edges[0, ol, 0]] for ol in odd_colors]
        clauses += [[-edges[0, ol, n-1]] for ol in odd_colors]
        
        return clauses


    def create_cnf(self, posnodes, negnodes, edges, parents, t_aux, m_aux, neg_aux, pos_dba, neg_dba
                , n, alphabet, nobfs, parity, reach):
        # we assume that 0 is the initial state and n-1 is the sink reject state
        clauses = self.make_deterministic_complete(edges, n, alphabet)
        # consistent with the samples on transitions
        sub_clauses = self.make_transiton_consistency(posnodes, negnodes, edges, n, alphabet, pos_dba, neg_dba)
        clauses = sub_clauses + clauses
        # consistent with positive samples

        sub_clauses = self.accept_positive_samples(posnodes, n, pos_dba)
        # consistent with negative samples
        clauses += sub_clauses
        sub_clauses = self.reject_negative_samples(negnodes, edges, neg_aux, reach, n, alphabet, neg_dba)
        clauses += sub_clauses
        # sys.exit(0)
        clauses += self.make_safe(edges, n, alphabet)
        # clauses = self.create_dfa_cnf(nodes, edges, input_sdfa,
        #                         n, alphabet)
        if not nobfs:
            sub_clauses = self.create_BFStree_cnf(edges, parents, t_aux, m_aux, n, alphabet, True)
            clauses = sub_clauses + clauses
        else:
            clauses += self.create_symmetry_break(edges, n, alphabet)   
        if parity:
            sub_clauses = self.create_DSA_cnf(edges, n, alphabet)
            clauses = sub_clauses + clauses

        #for cls in clauses:
        #    print(cls)
        return clauses


    def construct_dfa_from_model(self, model, edges, n, alphabet):
        # print("type(model) = " + str(type(model)))
        dfa = SDFA.sdfa()
        dfa.set_num_states(n)
        dfa.set_num_letters(len(alphabet))
        for p in range(n):
            print("State " + str(p))
            is_final = False
            if p != n-1:
                # print(" final")
                is_final = True
            
            for a, letter in enumerate(alphabet):
                # print("a =" +str(a) + ", letter=" + str(letter))
                for q in range(n):
                    if model[edges[p, a, q] - 1] > 0:
                        # print("var = " + str(e1[p, a, q]))
                        # print("letter " + str(letter) + " -> " + str(q))
                        dfa.add_transition(p, a, q)
            if is_final:
                dfa.add_final_state(p)
            #else:
            #    dfa.add_reject_state(p)

        dfa.add_initial_state(0)
        return dfa


    def solve(self,
            sat, n, alphabet, pos_dba, neg_dba, nobfs, parity, reach):

        accept_states = { s for s in range(neg_dba.num_states) if not (s in neg_dba.reject_states)}
        posnodes, negnodes, edges, parents, t_aux, m_aux, neg_aux = self.create_variables(n
                                                                    , alphabet
                                                                    , pos_dba.num_states
                                                                    , neg_dba.num_states
                                                                    , accept_states, reach)
        # solvers, Glucose3(), Cadical103(), Cadical153(), Gluecard4(), Glucose42()
        # g = Cadical153() #Lingeling() #Glucose42()
        # g = Glucose42()
        g = Solver(name=sat)

        clauses = self.create_cnf(posnodes, negnodes, edges, parents, t_aux, m_aux, neg_aux
                           , pos_dba, neg_dba, n, alphabet, nobfs, parity, reach)
        print("Created # of clauses: " + str(len(clauses)))
        for cls in clauses:
            # print(cls)
            g.add_clause(cls)

        is_sat = g.solve()
        #print(is_sat)

        if is_sat:
            print(g.get_model())
            print("Found a DFA with size " + str(n))
            model = g.get_model()
            for p in range(n-1):
                for q in range(n-1):
                    if reach:
                        print("[", p, ", ", q, "]=", neg_aux[p, q])
                        print( model[neg_aux[p,q] - 1] > 0)
            print("product")
            for p in range(neg_dba.num_states):
                for q in range(n):
                        print("[", p, ", ", q, "]=", negnodes[p, q])
                        print( model[negnodes[p,q] - 1] > 0)
            # now print out transition relation
            # print("type(model) = " + str(type(model)))
            dfa = self.construct_dfa_from_model(model, edges, n, alphabet)
            return (True, dfa)
        else:
            print("No DFA existing for size " + str(n))
            return (False, None)


    def minimise(self,
            pos_dba, neg_dba, sat, lbound, ubound, nobfs, parity, reach):

        assert pos_dba.num_letters == neg_dba.num_letters
        
        alphabet = list(range(pos_dba.num_letters))
        n = max(lbound, 2)
        # the bound should not be greater than the size of the input automata
        max_bound = min([pos_dba.num_states, ubound])
        max_bound = min([max_bound, neg_dba.num_states])
        # the maximal number of states must not be bigger than
        # the number of states in the input FA
        result_dfa = None
        print("Input positive DBA size: " + str(pos_dba.num_states))
        print("Input negative DBA size: " + str(neg_dba.num_states))

        print("Input alphabet size: " + str(pos_dba.num_letters))

        start_time = time.time()
        while n <= max_bound:
            print("Looking for DFA with " + str(n) + " states...")
            res, dfa = self.solve(sat, n, alphabet, pos_dba, neg_dba, nobfs, parity, reach)
            if res:
                result_dfa = dfa
                break
            else:
                n += 1
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 4)
        print(f"Elapsed time in minimiser: {elapsed_time} secs")
        return result_dfa



solver_choices = {"cadical195", "cadical103", "cadical153", "gluecard4", "glucose4",
                  "glucose42", "lingeling", "maplechrono", "mergesat3", "minisat22"}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extacting minimal safe DBA separating two input safe DBAs')
    parser.add_argument('--pos', metavar='path', required=True,
                        help='path to positive safe DBA')
    parser.add_argument('--neg', metavar='path', required=True,
                        help='path to negative safe DBA')
    parser.add_argument('--out', metavar='path', required=True,
                        help='path to output FA')
    parser.add_argument('--solver', type=str.lower, required=False,
                        choices=solver_choices, default="cadical195",
                        help='choose the SAT solver')
    parser.add_argument('--lower', type=int, required=False,
                        default=1,
                        help='the lower bound for the DFA')
    parser.add_argument('--upper', type=int, required=False,
                        default=sys.maxsize,
                        help='the upper bound for the DFA')
    parser.add_argument('--nobfs', action="store_true", required=False,
                        default=False,
                        help='disable the constraints for BFS tree')
    parser.add_argument('--parity', action="store_true", required=False,
                        default=False,
                        help='construct safety DBA for solving parity games')
    parser.add_argument('--reach', action="store_true", required=False,
                        default=False,
                        help='novel reachability relation encoding')
    args = parser.parse_args()
    minimiser = safe_miner() 
    pos = SDFA.sdfa()
    pos.load(args.pos)
    neg = SDFA.sdfa()
    neg.load(args.neg)
    with open("pos.dot", "w") as file:
        file.write(pos.dot())
    with open("neg.dot", "w") as file:
        file.write(neg.dot())
    print("Launch DBA minimiser...")
    dfa = minimiser.minimise(pos_dba=pos, neg_dba=neg, sat=args.solver
         , lbound=args.lower, ubound=args.upper, nobfs=args.nobfs, parity=args.parity, reach=args.reach)
    print("Output to " + args.out)
    with open(args.out, "w") as file:
        file.write(dfa.dot())
