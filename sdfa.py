import strunion
import copy
from hopcroft import sdfa_poly_minimiser
from acyclic import sdfa_acyclic_minimiser
import dfaminer

# separated DFA for the positive and negative samples
class sdfa:
    def __init__(self):
        self.num_letters = -1
        self.num_states = -1
        self.init_states = set()
        self.trans = []
        self.final_states = set()
        self.reject_states = set()
        self.rev_trans = None
        
    def add_initial_state(self, state):
        self.init_states.add(state)
    
    def set_num_letters(self, num_letters):
        self.num_letters = num_letters
                
    def set_num_states(self, num_states):
        self.num_states = num_states
        self.trans = [dict() for i in range(self.num_states)]
    
    def create_new_state(self):
        if self.num_states == -1:
            self.num_states = 0
        self.num_states += 1
        self.trans.append(dict())
        return self.num_states - 1

    def get_num_trans(self):
        return sum([len(state) for state in self.trans])

    def __verify_states__(self, state):
        return state < self.num_states and state >= 0
    
    def __verify_letters__(self, letter):
        return letter < self.num_letters and letter >= 0
        
    def add_transition(self, src, letter, dst):
        assert self.__verify_states__(src) and self.__verify_states__(dst)
        assert self.__verify_letters__(letter)
        self.trans[src][letter] = dst
    
    def add_final_state(self, state):
        assert self.__verify_states__(state)
        self.final_states.add(state)
    
    def add_reject_state(self, state):
        assert self.__verify_states__(state)
        self.reject_states.add(state)
        
    def run(self, state, word):
        curr_state = state
        for c in word:
            next_state = self.trans[curr_state].get(c, None)
            if next_state is None:
                return strunion.word_type.DONTCARE
            curr_state = next_state
        if curr_state in self.final_states:
            return strunion.word_type.ACCEPT
        elif curr_state in self.reject_states:
            return strunion.word_type.REJECT
        else:
            return strunion.word_type.DONTCARE
        
    @staticmethod    
    def combine(pos_dfa, neg_dfa, num_letters):
        res = sdfa()
        res.set_num_letters(num_letters)
        res.set_num_states(pos_dfa.num_states + neg_dfa.num_states)
        base = pos_dfa.num_states
        for init in pos_dfa.init_states:
            res.add_initial_state(init)
        for init in neg_dfa.init_states:
            res.add_initial_state(init + base)
        for fin in pos_dfa.final_states:
            res.add_final_state(fin)
        for fin in neg_dfa.final_states:
            res.add_reject_state(fin + base)
        # transitions
        for src, strans in enumerate(pos_dfa.trans):
            for _, (c, dst) in enumerate(strans.items()):
                res.add_transition(src, c, dst)
        for src, strans in enumerate(neg_dfa.trans):
            for _, (c, dst) in enumerate(strans.items()):
                res.add_transition(src + base, c, dst + base)
        
        return res
    
    def split(self):
        pos_dfa = sdfa()
        neg_dfa = sdfa()
        
        pos_dfa.set_num_letters(self.num_letters)
        neg_dfa.set_num_letters(self.num_letters)
        pos_dfa.set_num_states(self.num_states)
        neg_dfa.set_num_states(self.num_states)
        
        pos_dfa.init_states = self.init_states.copy()
        neg_dfa.init_states = self.init_states.copy()
        pos_dfa.final_states = self.final_states.copy()
        neg_dfa.final_states = self.reject_states.copy()
        
        pos_dfa.trans = copy.deepcopy(self.trans)
        neg_dfa.trans = copy.deepcopy(self.trans)
        
        return (pos_dfa, neg_dfa)
        
    def get_reachable_states(self):
        return self.__get_reachable_states(self.init_states, self.trans)
    
    def __get_reachable_states(self, inits, trans):
        reachable_states = set()
        reachable_states |= inits
        curr_set = set()
        curr_set |= inits
        while True:
            temp_set = set()
            for state in curr_set:
                for _, (key, next_state) in enumerate(trans[state].items()):
                    if isinstance(next_state, set): 
                        temp_set |= next_state
                    else:
                        temp_set.add(next_state)
            # now check newly reachable states
            temp_set -= reachable_states
            reachable_states |= temp_set
            
            if len(temp_set) == 0:
                break
            
            curr_set = temp_set
        
        return reachable_states
    
    def compute_reverse_transition(self):
        if self.rev_trans is None:
            self.rev_trans = [dict() for _ in range(self.num_states)]
            # obtain reverse transitions
            for state, strans in enumerate(self.trans):
                for _, (letter, next_state) in enumerate(strans.items()):
                    predecessors = self.rev_trans[next_state].get(letter, set())
                    predecessors.add(state)
                    self.rev_trans[next_state][letter] = predecessors
    
    def get_rev_rechable_states(self):
        # reachable from final and reject
        inits = self.final_states.copy()
        inits |= self.reject_states
        self.compute_reverse_transition()
        return self.__get_reachable_states(inits, self.rev_trans) 
    
    def reduce(self):
        reachable_states = self.get_reachable_states() & self.get_rev_rechable_states()
        state_map = { state : index for index, state in enumerate(reachable_states)}
        # print(state_map)
        result = sdfa()
        result.set_num_states(len(reachable_states))
        result.set_num_letters(self.num_letters)
        result.final_states = { state_map[state] for state in self.final_states & reachable_states}
        result.reject_states = { state_map[state] for state in self.reject_states & reachable_states }
        result.init_states = { state_map[state] for state in self.init_states & reachable_states}
        result.trans = [ dict( ) for i in range(result.num_states)]
        
        for state in reachable_states:
                for _, (letter, next_state) in enumerate(self.trans[state].items()):
                    if next_state in reachable_states:
                        result.add_transition(state_map[state], letter, state_map[next_state])
                        
        return result
    
    def minimise(self):
        poly_minimiser = sdfa_poly_minimiser(self)
        return poly_minimiser.minimise()
    
    def minimise_acyclic(self):
        acyclic_minimiser = sdfa_acyclic_minimiser(self)
        return acyclic_minimiser.minimise()
        
    def __str__(self):
        out_str = []
        out_str.append(str(self.num_states) + " " + str(self.num_letters) + "\n")
        for init in self.init_states:
            out_str.append("i " + str(init) + "\n")

        for src, strans in enumerate(self.trans):
            # print(src, strans)
            for _, (c, dst) in enumerate(strans.items()):
                # print(c, dst)
                out_str.append("t " + str(src) + " " + str(c) + " " + str(dst) + "\n")
        for fin in self.final_states:
            out_str.append("a " + str(fin) + "\n")
        for rej in self.reject_states:
            out_str.append("r " + str(rej))
        return "".join(out_str)
    
    def textual(self): 
        return self.__str__()

    def dot(self, alphabet):
        str_list = []
        str_list.append("digraph {\n")
        # str_list.append("  rankdir=LR;\n")
        str_list.append("  rankdir=TB;\n")
        for s in range(self.num_states):
            shape = ", shape = circle"
            if s in self.final_states:
                shape = ", shape = doublecircle"
            elif s in self.reject_states:
                shape = ", shape = square"
            str_list.append(" " + str(s) + ' [label= "' + str(s) + '"' + shape + "];\n")
        for src, strans in enumerate(self.trans):
            for _, (c, dst) in enumerate(strans.items()):
                str_list.append(
                    " " + str(src) + " -> " + str(dst) + ' [label="' + str(alphabet[c]) + '"];\n'
                )

        num = 0
        for init_state in self.init_states:
            str_list.append(
                " " + str(self.num_states + num) + ' [label="", shape = plaintext];\n'
            )
            str_list.append(
                " "
                + str(self.num_states + num)
                + " -> "
                + str(init_state)
                + ' [label=""];\n'
            )
            num += 1
        str_list.append("}\n")
        return "".join(str_list)

    def load(self, file_name):
        bufsize = 65536
        num_line = 0
        left = set()
        right = set()
        with open(file_name) as infile:
            # open this file
            while True:
                lines = infile.readlines(bufsize)
                # ignore empty lines
                if not lines:
                    break
                for line in lines:
                    line_brk = line.split()
                    if len(line_brk) == 0:
                        # ignore lines with only spaces
                        continue
                    if num_line == 0:
                        self.num_states = int(line_brk[0])
                        self.num_letters = int(line_brk[1])
                        # print("#S = " + str(num_states))
                        # print("#C = " + str(num_colors))
                        self.trans = [dict() for i in range(self.num_states)]
                        num_line += 1
                    elif num_line == 1 and line[0] != "i":
                        # old format
                        init_state = int(line_brk[0])
                        self.add_initial_state(init_state)
                        num_line += 1
                    elif line[0] == "i":
                        init_state = int(line_brk[1])
                        # print("init = " + str(init_state))
                        self.add_initial_state(init_state)
                        num_line += 1
                    elif line[0] == "a":
                        state = int(line_brk[1])
                        # print("acc = " + str(state))
                        self.add_final_state(state)
                    elif line[0] == "r":
                        state = int(line_brk[1])
                        # print("rej = " + str(state))
                        self.add_reject_state(state)
                    elif line[0] == "t":
                        # now it is after three
                        src_state = int(line_brk[1])
                        letter = int(line_brk[2])
                        dst_state = int(line_brk[3])
                        # right.add(dst_state)
                        # left.add(src_state)
                        # print("src = " + str(src_state) + " letter = " +
                        #   str(letter) + " dst = " + str(dst_state))
                        self.add_transition(src_state, letter, dst_state)
        # result = left - right
        # print(result)
        # sys.exit(1)


#sd = sdfa()
#sd.load("data2-3-all-dfa.txt")
#print(sd)
#print(sd.dot())
