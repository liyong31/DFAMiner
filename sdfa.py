import functools

# separated DFA for the positive and negative samples
class sdfa:
        
    def __init__(self):
        self.num_letters = -1
        self.num_states = -1
        self.init_states = set()
        self.trans = []
        self.final_states = set()
        self.reject_states = set()

    def __str__(self):
        out_str = []
        out_str.append(str(self.num_states) + " " + str(self.num_letters) + "\n")
        for init in self.init_states:
            out_str.append( "i " + str(init) + "\n")

        for src, strans in enumerate(self.trans):
            # print(src, strans)
            for _, (c, dst) in enumerate(strans.items()):
                # print(c, dst)
                out_str.append("t " + str(src) + " " + str(c) + " " + str(dst) + "\n")
        for fin in self.final_states:
            out_str.append("a " + str(fin) + "\n")
        for rej in self.reject_states:
            out_str.append("r " + str(rej))
        return ''.join(out_str)
    
    def dot(self):
        str_list = []
        str_list.append("digraph {\n")
        str_list.append("  rankdir=LR;\n")
        for s in range(self.num_states):
            shape = ", shape = circle" 
            if s in self.final_states:
                shape = ", shape = doublecircle"
            elif s in self.reject_states:
                shape = ", shape = square"
            str_list.append(" " + str(s) + " [label= \"" + str(s) + "\"" + shape + "];\n")
        for src, strans in enumerate(self.trans):
            for _, (c, dst) in enumerate(strans.items()):
                str_list.append(" " + str(src) + " -> " + str(dst) + " [label=\"" + str(c) + "\"];\n")
        
        num = 0
        for init_state in self.init_states:
            str_list.append("  " + str(self.num_states + num) + " [label=\"\", shape = plaintext];\n")
            str_list.append("  " + str(self.num_states + num) + " -> " + str(init_state) + " [label=\"\"];\n")
            num += 1
        str_list.append("}\n")
        return ''.join(str_list)    

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
                        self.init_states.add(init_state)
                        num_line += 1
                    elif line[0] == "i":
                        init_state = int(line_brk[1])
                        # print("init = " + str(init_state))
                        self.init_states.add(init_state)
                        num_line += 1
                    elif line[0] == "a":
                        state = int(line_brk[1])
                        # print("acc = " + str(state))
                        self.final_states.add(state)
                    elif line[0] == "r":
                        state = int(line_brk[1])
                        # print("rej = " + str(state))
                        self.reject_states.add(state)
                    elif line[0] == "t":
                        # now it is after three
                        src_state = int(line_brk[1])
                        letter = int(line_brk[2])
                        dst_state = int(line_brk[3])
                        right.add(dst_state)
                        left.add(src_state)
                        # print("src = " + str(src_state) + " letter = " +
                        #   str(letter) + " dst = " + str(dst_state))
                        self.trans[src_state][letter] = dst_state
        result = left - right
        print(result)
        # sys.exit(1)

sd = sdfa()
sd.load("data2-3-all-dfa.txt")
print(sd)
print(sd.dot())