import strunion as SU
import sdfa as SDFA
import hopcroft

# minimiser for acylic sdfa
class sdfa_acyclic_minimiser:

    def __init__(self, sdfa):
        self.sdfa = sdfa.reduce()
        # transitions of state_t
        self.states = []
        # transitions of state_t to its partition id
        self.register = {}
        # original state id to its partition id
        self.state_map = {}

    ## minimise deterministic acyclic automata, so we start from leaf states
    def minimise(self):
        # Step 1. prepare the DAG/ acyclic structure
        pred_map = {}
        worklist = []
        succ_map = {}
        for state in range(self.sdfa.num_states):
            succs = set()
            for symbol in self.sdfa.trans[state].keys():
                pred_set = pred_map.get(self.sdfa.trans[state][symbol], set())
                pred_set.add(state)
                pred_map[self.sdfa.trans[state][symbol]] = pred_set

                succs.add(self.sdfa.trans[state][symbol])

            succ_map[state] = succs
            if not succs:
                worklist.append(state)

        # Step 3. recursively remove successors and add leaf nodes
        while len(worklist) > 0:
            # each state will only be processed once
            curr_state = worklist.pop(0)
            repr_state = self.__register_new_state(curr_state)
            self.state_map[curr_state] = repr_state
            # in case state has no predecessors
            preds = pred_map.get(curr_state, set())
            # Step 4. remove this state from succ_map
            for pred in preds:
                succ_set = succ_map.get(pred, None)
                if succ_set is None:
                    # already processed
                    continue
                succ_set.remove(curr_state)
                if not succ_set:
                    # pred is leaf node now
                    worklist.append(pred)
                    # remove this key as it will not be used
                    succ_map.pop(pred)
                else:
                    succ_map[pred] = succ_set

        return hopcroft.sdfa_poly_minimiser.build_minimised_dfa(
            self.sdfa, len(self.states), self.state_map
        )
        # return self.__build_minimised_sdfa()

    def __register_new_state(self, curr_state):
        # 1. get successor information
        # invariant: by default, if pred is here, then all its successors
        # must have been processed
        state = SU.dfa_builder.state_t()
        for c in range(self.sdfa.num_letters):
            succ = self.sdfa.trans[curr_state].get(c, None)
            if succ is not None:
                state.labels.append(c)
                state.states.append(self.states[self.state_map[succ]])
        # acceptance type
        if curr_state in self.sdfa.final_states:
            state.is_final = SU.word_type.ACCEPT
        elif curr_state in self.sdfa.reject_states:
            state.is_final = SU.word_type.REJECT
        else:
            state.is_final = SU.word_type.DONTCARE

        # now search for equivalence classes
        repr_state = self.register.get(state, None)
        if repr_state is not None:
            return repr_state
        else:
            repr_state = len(self.states)
            self.states.append(state)
            # self.state_map[curr_state] = repr_state
            self.register[state] = repr_state
        return repr_state
    
    def __build_minimised_sdfa(self):
        result = SDFA.sdfa()
        result.set_num_states(len(self.states))
        result.set_num_letters(self.sdfa.num_letters)
        
        worklist = [ self.state_map[state] for state in self.sdfa.init_states]
        already_added = set()
        for state in worklist:
            result.add_initial_state(state)
            already_added.add(state)
        
        while len(worklist) > 0:
            # travserse states one by one
            curr_state_id = worklist.pop(0)
            curr_state = self.states[curr_state_id]
            
            # acceptance
            if curr_state.is_final == SU.word_type.ACCEPT:
                result.add_final_state(curr_state_id)
            elif curr_state.is_final == SU.word_type.REJECT:
                result.add_reject_state(curr_state_id)
            
            # transitions
            for letter, dest in zip(curr_state.labels, curr_state.states):
                dest_state_id = self.register[dest]
                result.add_transition(curr_state_id, letter, dest_state_id)
                if dest_state_id in already_added:
                    continue
                else:
                    already_added.add(dest_state_id)
                    worklist.append(dest_state_id)
                    
        return result