"""Full minimisation benchmark: start at 1 or certified clique lower bound.

All competing strategies use:
  - the SAME compact eager CNF (without implied positive/reject pair product);
  - the SAME target-BFS symmetry constraints;
  - an independent, fresh CaDiCaL solver for each attempted target size.

Configurations differ ONLY in whether they collect witnessed incompatibility
information, add 300 selected scored pair lemmas, and/or skip sizes below
the witnessed clique lower bound. No oracle-optimum starting size is used.

A candidate is checked against the ORIGINAL 3DFA by complete product
verification. Every smaller tested size must have returned UNSAT. A
skipped size is justified by a separately validated clique certificate.
Known reference optima are checked only AFTER solving.

The per-configuration timeout includes 3DFA construction, preprocessing,
ALL attempted target sizes, model verification and final reporting. All
completed per-k results are retained even on timeout/failure.

Examples:
 python benchmarks/compare_full_minimisation.py --cases repo8 repo10
 python benchmarks/compare_full_minimisation.py --cases sparse9600 --timeout 80
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.compare_large_3dfa import prepare, KNOWN_REPO_OPT
from cegar_minimiser import sdfa_cegar_minimiser
from hybrid_eager import HybridEagerMinimiser
from pysat.solvers import Solver

MODES = {
    # name:         use sound LB, pair budget, pair scoring
    'compact-bfs':       (False, 0,   'degree'),
    'bound-bfs':         (True,  0,   'degree'),
    'degree-from1':      (False, 300, 'degree'),
    'degree-bound':      (True,  300, 'degree'),
    'short-bound':       (True,  300, 'short'),
    'impact-bound':      (True,  300, 'impact'),
}
# Expected optima are for validation, never used to choose starting k.
EXPECTED_OPTIMUM = {**KNOWN_REPO_OPT}


def run_worker(case, mode, max_k=15):
    if mode not in MODES:
        raise ValueError('Unknown benchmark mode')
    use_bound, pair_budget, pair_policy = MODES[mode]
    started = perf_counter()
    machine, input_meta = prepare(case)
    input_seconds = perf_counter() - started
    alphabet = list(range(machine.num_letters))
    engine = HybridEagerMinimiser(
        pair_budget=pair_budget,
        policy=pair_policy,
        bound_budget_ms=(25 if (use_bound or pair_budget) else 0),
        pair_budget_ms=15,
        max_candidates=256,
        max_suffixes=64,
        max_depth=9,
        explicit_label_pairs=False,
        verbose=False)
    pre_start = perf_counter()
    engine.prepare(machine)
    preprocess_seconds = perf_counter() - pre_start
    certified_bound = engine.lower_bound
    if engine.certificate:
        # Verify every clique witness and every additional propagation lemma.
        assert engine.certificate.verify()
        assert engine.certificate.verify_scored_pairs(engine.selected_pairs)
    first_k = certified_bound if use_bound else 1
    assert first_k >= 1
    assert first_k <= max_k
    print('BUILT ' + json.dumps(
        dict(case=case,mode=mode,input_states=machine.num_states,
             input_edges=input_meta['input_edges'],first_k=first_k,
             certified_lower_bound=certified_bound,
             skipped_k=list(range(1,first_k)),
             selected_pairs=len(engine.selected_pairs),
             input_seconds=round(input_seconds,5),
             preprocessing_seconds=round(preprocess_seconds,5)),
        sort_keys=True),flush=True)

    outcomes=[]
    optimum=None
    for k in range(first_k,max_k+1):
        build_start=perf_counter()
        variables=engine.create_variables(k,alphabet,machine.num_states)
        clauses=engine.create_cnf(
            *variables,machine,k,alphabet,False,False) # target BFS ON
        cnf_seconds=perf_counter()-build_start

        boot_start=perf_counter()
        solver=Solver(name='cadical195',bootstrap_with=clauses)
        bootstrap_seconds=perf_counter()-boot_start
        try:
            sat_start=perf_counter()
            is_sat=solver.solve()
            sat_seconds=perf_counter()-sat_start
            stats=solver.accum_stats()
            if is_sat:
                candidate=engine.construct_dfa_from_model(
                    solver.get_model(),variables[1],k,alphabet)
                bad,_=sdfa_cegar_minimiser._verify_product(
                    machine,candidate,1)
                if bad:
                    raise AssertionError(('Incorrect SAT DFA',case,mode,k,bad))
                optimum=k
        finally:
            solver.delete()

        step=dict(k=k,status='SAT' if is_sat else 'UNSAT',
                  clauses=len(clauses),cnf_seconds=round(cnf_seconds,5),
                  bootstrap_seconds=round(bootstrap_seconds,5),
                  sat_seconds=round(sat_seconds,5),
                  conflicts=stats.get('conflicts'),
                  decisions=stats.get('decisions'),
                  propagations=stats.get('propagations'))
        outcomes.append(step)
        print('STEP ' + json.dumps(
            dict(case=case,mode=mode,**step),sort_keys=True),flush=True)
        if is_sat:
            break

    if optimum is None:
        raise AssertionError(('Could not find a solution below max_k',case,mode,max_k))
    assert all(x['status']=='UNSAT' for x in outcomes[:-1])
    if case in EXPECTED_OPTIMUM:
        assert optimum==EXPECTED_OPTIMUM[case], (case,mode,optimum)
    # The generator has a verified six-state source DFA. Here the
    # SAT proof at k=5, or a valid clique witness of size 6, is what
    # establishes that the first SAT at k=6 is minimal.
    if case.startswith('sparse'):
        assert optimum==6,(case,mode,optimum)

    totals=dict(
        input_seconds=round(input_seconds,5),
        preprocessing_seconds=round(preprocess_seconds,5),
        cnf_seconds=round(sum(x['cnf_seconds'] for x in outcomes),5),
        bootstrap_seconds=round(sum(x['bootstrap_seconds'] for x in outcomes),5),
        sat_seconds=round(sum(x['sat_seconds'] for x in outcomes),5),
    )
    result=dict(case=case,mode=mode,status='OPTIMUM_CERTIFIED',
                states=machine.num_states,
                input_edges=input_meta['input_edges'],
                first_k=first_k,certified_lower_bound=certified_bound,
                skipped_k=list(range(1,first_k)),
                selected_pairs=len(engine.selected_pairs),
                **totals,
                optimal_k=optimum,
                attempted_k=[x['k'] for x in outcomes],
                sat_calls=len(outcomes),
                total_conflicts=sum(x['conflicts'] or 0 for x in outcomes),
                total_seconds=round(perf_counter()-started,5),
                peak_rss_mb=round(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss/1024,2),
                per_k=outcomes)
    print('SUMMARY ' + json.dumps(result,sort_keys=True),flush=True)


def run_parent(args):
    results=[]
    for case in args.cases:
        for repetition in range(args.repeats):
            # Rotate execution order across repetitions to reduce bias.
            shift=repetition%len(args.modes)
            modes=args.modes[shift:]+args.modes[:shift]
            best={}
            for mode in modes:
                start=perf_counter()
                command=[sys.executable,'-u',str(Path(__file__).resolve()),
                         '--worker',case,mode,'--max-k',str(args.max_k)]
                try:
                    proc=subprocess.run(
                        command,cwd=ROOT,capture_output=True,text=True,
                        timeout=args.timeout,
                        env={**os.environ,'PYTHONHASHSEED':'0'})
                    lines=proc.stdout.splitlines()
                    summaries=[json.loads(line[8:]) for line in lines
                               if line.startswith('SUMMARY ')]
                    steps=[json.loads(line[5:]) for line in lines
                           if line.startswith('STEP ')]
                    built=[json.loads(line[6:]) for line in lines
                           if line.startswith('BUILT ')]
                    if proc.returncode or len(summaries)!=1:
                        row=dict(case=case,mode=mode,
                                 status='ERROR',
                                 wall_seconds=round(perf_counter()-start,4),
                                 partial_steps=steps,partial_build=built,
                                 error=(proc.stderr+proc.stdout)[-1800:])
                    else:
                        row=summaries[0]
                        row['wall_seconds']=round(perf_counter()-start,4)
                        best[mode]=row['optimal_k']
                except subprocess.TimeoutExpired as error:
                    stdout=error.stdout or b''
                    if isinstance(stdout,bytes):
                        stdout=stdout.decode('utf8','replace')
                    row=dict(case=case,mode=mode,
                             status='TIMEOUT',
                             wall_seconds=args.timeout,
                             partial_steps=[json.loads(line[5:])
                                            for line in stdout.splitlines()
                                            if line.startswith('STEP ')],
                             partial_build=[json.loads(line[6:])
                                            for line in stdout.splitlines()
                                            if line.startswith('BUILT ')])
                row['repeat']=repetition
                results.append(row)
                print('RESULT '+json.dumps(row,sort_keys=True),flush=True)
                Path(args.output).write_text(json.dumps(results,indent=2))
            if len(set(best.values()))>1:
                raise AssertionError(('Conflicting optimum',case,repetition,best))

    summary=dict(cases=args.cases,modes=args.modes,repeats=args.repeats,
                 successes=sum(x['status']=='OPTIMUM_CERTIFIED' for x in results),
                 timeouts=sum(x['status']=='TIMEOUT' for x in results),
                 errors=sum(x['status']=='ERROR' for x in results))
    print('COMPLETE '+json.dumps(summary,sort_keys=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker',nargs=2)
    ap.add_argument('--cases',nargs='+',
                    default=['repo7','repo8','repo10','sparse2400','sparse4800'])
    ap.add_argument('--modes',nargs='+',choices=list(MODES),
                    default=['compact-bfs','bound-bfs',
                             'degree-from1','degree-bound','short-bound'])
    ap.add_argument('--max-k',type=int,default=12)
    ap.add_argument('--timeout',type=int,default=90)
    ap.add_argument('--repeats',type=int,default=1)
    ap.add_argument('--output',default='full_minimisation_results.json')
    args=ap.parse_args()
    if args.worker:
        run_worker(args.worker[0],args.worker[1],max_k=args.max_k)
    else:
        run_parent(args)
