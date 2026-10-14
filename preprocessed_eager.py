"""Experimental eager DFA minimiser with source preprocessing and RGS symmetry."""
from eager_cnf_variants import ReducedEagerMinimiser
from preprocess_3dfa import trim_observation_irrelevant,rgs_symmetry_clauses


class PreprocessedEagerMinimiser(ReducedEagerMinimiser):
    def __init__(self,mode='both',trim=True,symmetry='none',prefix=32):
        super().__init__(mode)
        self.trim=trim
        self.symmetry=symmetry
        self.prefix=prefix

    def preprocess(self,machine):
        if self.trim:
            return trim_observation_irrelevant(machine)
        return machine,dict(kept_states=machine.num_states,
                            removed_states=0,elapsed_seconds=0)

    def create_cnf(self,nodes,edges,parents,t_aux,m_aux,
                   input_sdfa,n,alphabet,nobfs,safety):
        result=super().create_cnf(nodes,edges,parents,t_aux,m_aux,
                                  input_sdfa,n,alphabet,nobfs,safety)
        if self.symmetry!='none':
            if not nobfs:
                raise ValueError('Do not combine input-RGS with target BFS')
            result.extend(rgs_symmetry_clauses(
                nodes,input_sdfa,n,self.symmetry,self.prefix))
        return result
