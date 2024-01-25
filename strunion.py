from typing import List
from collections import defaultdict
from bisect import bisect_left
from enum import Enum

import sdfa as SDFA
import DFAgen

class word_type(Enum):
        DONTCARE = 0
        REJECT = 1
        ACCEPT = 2
        
class dfa_builder:
    LEXICOGRAPHIC_ORDER = lambda s1, s2: (
         len(s1) - len(s2)
    ) if len(s1) != len(s2) else (
        -1 if s1 < s2 else (1 if s1 > s2 else 0)
    )
    


    class state_t:

        def __init__(self):
            self.labels = []
            self.states = []
            self.is_final = word_type.DONTCARE

        def get_state(self, label):
            index = bisect_left(self.labels, label)
            return self.states[index] if index >= 0 and index < len(self.labels) and self.labels[index] == label else None

        def get_transition_labels(self):
            return self.labels

        def get_states(self):
            return self.states

        def __eq__(self, other):
            return self.is_final == other.is_final and self.labels == other.labels and self.reference_equals(self.states, other.states)

        def has_children(self):
            return len(self.labels) > 0

        def is_final_state(self):
            return self.is_final

        def __hash__(self):
            hash_val = 1 if self.is_final else 0
            hash_val ^= hash_val * 31 + len(self.labels)
            for lab in self.labels:
                hash_val ^= hash_val * 31 + lab
            for st in self.states:
                hash_val ^= id(st)
            return hash_val

        def new_state(self, label):
            assert label not in self.labels, f"State already has transition labeled: {label}"
            self.labels.append(label)
            new_state = dfa_builder.state_t()

            self.states.append(new_state)
            return new_state

        def last_child(self):
            assert self.has_children(), "No outgoing transitions."
            return self.states[-1]

        def last_child_with_label(self, label):
            index = len(self.labels) - 1
            return self.states[index] if index >= 0 and self.labels[index] == label else None

        def replace_last_child(self, new_state):
            assert self.has_children(), "No outgoing transitions."
            self.states[-1] = new_state

        @staticmethod
        def copy_of(original, new_length):
            return original[:new_length]

        @staticmethod
        def reference_equals(a1, a2):
            return len(a1) == len(a2) and all(x is y for x, y in zip(a1, a2))

    def __init__(self):
        self.register = {} #defaultdict(lambda: None)
        self.root = self.state_t()
        self.previous = None
        
    def size(self):
        return len(self.register)

    def add(self, current, type):
        assert self.register is not None, "Automaton already built."
        assert len(current) > 0, "Input sequences must not be empty."
        assert self.previous is None or dfa_builder.LEXICOGRAPHIC_ORDER(self.previous, current) <= 0, f"Input must be sorted: {self.previous} >= {current}"
        self.set_previous(current)

        pos = 0
        max_len = len(current)
        state = self.root
        next_state = None
        while pos < max_len:
            # print("pos: ", pos, " letter: ", current[pos])
            next_state = state.last_child_with_label(current[pos])
            if next_state is None:
                break
            state = next_state
            pos += 1

        if state.has_children():
            self.replace_or_register(state)

        self.add_suffix(state, current, pos, type)

    def complete(self):
        if self.register is None:
            raise RuntimeError("Automaton already built.")

        if self.root.has_children():
            self.replace_or_register(self.root)

        self.register = None
        return self.root

    @staticmethod
    def convert(state, dfa, visited):
        # reference the value by id
        src = visited.get(id(state), None)
        if src is not None:
            return src

        src = dfa.create_new_state()
        #src = len(visited)
        if state.is_final == word_type.ACCEPT:
            dfa.add_final_state(src)
        elif state.is_final == word_type.REJECT:
            dfa.add_reject_state(src)

        visited[id(state)] = src
        for label, target in zip(state.labels, state.states):
            dst = dfa_builder.convert(target, dfa, visited)
            dfa.add_transition(src, label, dst)

        return src

    @staticmethod
    def build(builder, num_letters):
        state_map = {}
        # create SDFA instance
        dfa = SDFA.sdfa()
        dfa.set_num_letters(num_letters)
        
        state = builder.complete()
        init = dfa_builder.convert(state, dfa, state_map)
        dfa.add_initial_state(init)
        return dfa

    def set_previous(self, current):
        if self.previous is None:
            self.previous = ""

        self.previous = current

    def replace_or_register(self, state):
        child = state.last_child()
        if child.has_children():
            self.replace_or_register(child)

        registered = self.register.get(child, None)
        if registered is not None:
            state.replace_last_child(registered)
        else:
            self.register[child] = child

    def add_suffix(self, state, current, from_index, type):
        for char in current[from_index:]:
            state = state.new_state(char)

        state.is_final = type
'''
num_colors = 4
length = 7
pos, negs, all_vals, dontcares = DFAgen.get_all_data(num_colors, length)
builder = dfa_builder()

builderSDFA = dfa_builder()

for val in all_vals:
    print(val)
    if val[1] == 1:
        builderSDFA.add(val[0], word_type.ACCEPT)
    else:
        builderSDFA.add(val[0], word_type.REJECT)
sdfa_in = dfa_builder.build(builderSDFA, num_colors)


for sample in pos:
    print(sample, word_type.ACCEPT)
    builder.add(sample, word_type.ACCEPT)
    
dfa = dfa_builder.build(builder, num_colors)
pos_num = dfa.num_states
print("positive:\n", dfa.dot())


builder = dfa_builder()
for sample in negs:
    print(sample)
    builder.add(sample, word_type.ACCEPT)
dfa = dfa_builder.build(builder, num_colors)
print("negative:\n", dfa.dot())
print("3dfa:\n",sdfa_in.dot())


print("#pos states: ", pos_num)
print("#neg states: ", dfa.num_states)
print("#3dfa: ", sdfa_in.num_states)

'''