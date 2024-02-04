import sdfa as SDFA

class sdfa_poly_minimiser:
    
    def __init__(self, sdfa):
        self.sdfa = sdfa.reduce()
        self.partition_queue = []
        self.splitter_ids = []
        self.state_map = dict()
        
    
    def minimise(self):
        # Step 1: Split states into accepting, don't care and non-accepting sets
        self.partition_queue.append(set())
        self.splitter_ids.append(0)
        has_reject_states = len(self.sdfa.reject_states) > 0
        has_final_states = len(self.sdfa.final_states) > 0
        partition_id = 1
        final_partition_id = -1
        reject_partition_id = -1
        if has_final_states:
            self.splitter_ids.append(partition_id)
            self.partition_queue.append(set())
            final_partition_id = partition_id
            partition_id += 1
        if has_reject_states:
            self.partition_queue.append(set())
            self.splitter_ids.append(partition_id)
            reject_partition_id = partition_id
        
        # Step 2: Initialize the partition queue
        for state in range(self.sdfa.num_states):
            if state in self.sdfa.final_states:
                self.partition_queue[final_partition_id].add(state)
                self.state_map[state] = final_partition_id
            elif state in self.sdfa.reject_states:
                self.partition_queue[reject_partition_id].add(state)
                self.state_map[state] = reject_partition_id
            else:
                self.partition_queue[0].add(state)
                self.state_map[state] = 0 

        self.sdfa.compute_reverse_transition()
        # Step 3: Refine the partition until it no longer changes
        while len(self.splitter_ids) > 0:
            # add this partition as splitter?
            split_block_id = self.splitter_ids.pop(0)
            split_block = self.partition_queue[split_block_id]
            self.__refine_partition(split_block)
        
        # Step 4: Build the minimized DFA
        return sdfa_poly_minimiser.build_minimised_dfa(self.sdfa, len(self.partition_queue), self.state_map)

    # algorithm from https://en.wikipedia.org/wiki/DFA_minimization
    def __refine_partition(self, split_block):
        symbols = set()
        
        # used symbols
        for state in split_block:
            for symbol in self.sdfa.rev_trans[state].keys():
                symbols.add(symbol)

        for symbol in symbols:
            X = set()
            # compute all states that reach a state in splitter on symbol
            for state in split_block:
                prev_state = self.sdfa.rev_trans[state].get(symbol, None)
                if prev_state is not None:
                    X |= prev_state
            if not X:
                continue
            # refine the set Y in partition_queue that 
            # X /\ Y and Y \ X is not empty
            # shallow copy partitions
            tmp = self.partition_queue.copy()
            
            for p_id, Y in enumerate(self.partition_queue):
                inter_set = X & Y
                minus_set = Y - X

                # if one of them is empty, continue
                if not minus_set or not inter_set:
                    continue
                
                # change current Y to inter_set
                tmp[p_id] = inter_set
                ext_id = len(tmp)
                tmp.append(minus_set)
                
                if p_id in self.splitter_ids:
                    # keep p_id and add ext_id
                    self.splitter_ids.append(ext_id)
                else:
                    # make sure it is nlogn
                    if len(inter_set) <= len(minus_set):
                        self.splitter_ids.append(p_id)
                    else:
                        self.splitter_ids.append(ext_id)
                # update partition id for states
                for state in minus_set:
                    self.state_map[state] = ext_id
                        
            self.partition_queue = tmp
    
    @staticmethod
    def build_minimised_dfa(sdfa, num_partitions, state_map):
        result = SDFA.sdfa()
        result.set_num_states(num_partitions)
        result.set_num_letters(sdfa.num_letters)
        
        # transitions
        for src, strans in enumerate(sdfa.trans):
            # print(src, strans)
            for _, (c, dst) in enumerate(strans.items()):
                result.add_transition(state_map[src], c, state_map[dst])
                
            # acceptance and initials
            if src in sdfa.init_states:
                result.add_initial_state(state_map[src])
            if src in sdfa.final_states:
                result.add_final_state(state_map[src])
            if src in sdfa.reject_states:
                result.add_reject_state(state_map[src])
        
        return result
