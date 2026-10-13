"""Measure exact incremental incompatibility collection and clique quality."""
import json
import sys
from pathlib import Path
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarks.compare_large_3dfa import read_samples, make_sparse
from strunion import dfa_builder, word_type


def build(alphabet,samples,collect):
    b=dfa_builder(collect_incompatibility=collect)
    for word,sign in sorted(samples):
        if word:
            b.add(word,word_type.ACCEPT if sign>0 else word_type.REJECT)
    return dfa_builder.build(b,alphabet)

for name in ('repo4','repo7','repo8','repo10','repo12','sparse600','sparse2400'):
    alphabet,samples=(read_samples(name) if name.startswith('repo') else make_sparse(name))
    start=perf_counter()
    baseline=build(alphabet,samples,False)
    normal=perf_counter()-start
    start=perf_counter()
    collected=build(alphabet,samples,True)
    extra=perf_counter()-start
    info=collected.incompatibility
    assert info.verify()
    assert collected.trans==baseline.trans
    assert collected.final_states==baseline.final_states
    assert collected.reject_states==baseline.reject_states
    print('COLLECT',json.dumps(dict(case=name,states=baseline.num_states,
          edges=sum(map(len,baseline.trans)),base_seconds=round(normal,4),
          collected_seconds=round(extra,4),clique=len(info.clique),
          pairs=sum(row.bit_count() for row in info.adjacency)//2,
          checks=info.pairs_checked)),flush=True)
