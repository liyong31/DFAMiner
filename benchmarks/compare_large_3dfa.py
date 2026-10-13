"""Larger 3DFA comparison on repository samples and generated sparse samples.

Run a particular case (native PySAT/CaDiCaL 1.9.5 required):
  python benchmarks/compare_large_3dfa.py --cases repo7 repo10 sparse600
  python benchmarks/compare_large_3dfa.py --cases sparse2400 --seconds 60

Every solver is launched as a *separate subprocess* with a strict timeout,
peak RSS measurement and an independent full-product candidate check. This
prevents a stalled UNSAT proof from blocking all subsequent cases. Results
are JSON plus a concise human-readable table.

The generator builds a minimised acyclic 3DFA using the repository's own
lexicographic incremental builder. Sparse labels come from a six-state
modular counter, ensuring that a small consistent target DFA exists. Its
minimum relative to the finite sample is checked, not assumed.
"""

import argparse
import json
import os
from pathlib import Path
import random
import resource
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sdfa import sdfa
import strunion
from minimiser import sdfa_minimiser
from cegar_minimiser import sdfa_cegar_minimiser
from cdcl_product import CDCLProductMinimiser

DEFAULT_MODES = ('eager', 'cegar', 'cdcl', 'rollback')
KNOWN_REPO_OPT = {'repo4': 4, 'repo7': 7, 'repo8': 8, 'repo10': 9, 'repo12': 12}


def read_samples(case):
    n = int(case.removeprefix('repo'))
    filename = ROOT / 'DFA_Miner_dataset' / 'dataset' / str(n) / ('res%d-1.txt' % n)
    if not filename.is_file():
        raise FileNotFoundError(str(filename))
    with open(filename) as handle:
        header = next(handle).split()
        declared, alphabet = (int(v) for v in header[:2])
        samples = []
        for line in handle:
            fields = line.split()
            if not fields or fields[0] == '-1':
                continue
            label, length = int(fields[0]), int(fields[1])
            word = tuple(int(v) for v in fields[2:])
            if len(word) != length:
                raise ValueError('Incorrect sample length in ' + str(filename))
            samples.append((word, 1 if label else -1))
    if len(samples) != declared:
        raise AssertionError((filename, len(samples), declared))
    return alphabet, samples


def make_sparse(case):
    # Eg sparse1800: 1,800 random labelled words over {0,1};
    # lengths vary between 12 and 20. Distinct words are labelled
    # by the same six-state modular counter.
    count = int(case.removeprefix('sparse'))
    rng = random.Random(20261013 + count)
    used = {}
    while len(used) < count:
        length = rng.randrange(12, 21)
        word = tuple((rng.getrandbits(1)) for _ in range(length))
        number_ones = sum(word)
        positive = number_ones % 6 < 3
        used[word] = 1 if positive else -1
    return 2, sorted(used.items())


def build_3dfa(alphabet, samples):
    builder = strunion.dfa_builder()
    unique = {}
    for word, label in samples:
        if word in unique and unique[word] != label:
            raise ValueError('Inconsistent duplicate labelled word: ' + repr(word))
        unique[word] = label
    empty = unique.pop((), None)
    for word, label in sorted(unique.items()):
        builder.add(word, strunion.word_type.ACCEPT if label == 1
                    else strunion.word_type.REJECT)
    automaton = strunion.dfa_builder.build(builder, alphabet)
    if empty:
        for root in automaton.init_states:
            if empty == 1:
                automaton.add_final_state(root)
            else:
                automaton.add_reject_state(root)
    # Verify correctness against the *original* labelled words, independently
    # of minimisation solver output. No contradictory classification allowed.
    for word, label in samples:
        for root in automaton.init_states:
            actual = automaton.run(root, word)
            expected = (strunion.word_type.ACCEPT if label == 1
                        else strunion.word_type.REJECT)
            if actual != expected:
                raise AssertionError((word, actual, expected))
    return automaton


def prepare(case):
    start = time.perf_counter()
    alphabet, samples = (read_samples(case) if case.startswith('repo')
                         else make_sparse(case))
    automaton = build_3dfa(alphabet, samples)
    stats = dict(case=case,
                 samples=len(samples),
                 input_states=automaton.num_states,
                 input_edges=sum(len(t) for t in automaton.trans),
                 input_positive_states=len(automaton.final_states),
                 input_negative_states=len(automaton.reject_states),
                 build_seconds=round(time.perf_counter() - start, 4))
    return automaton, stats


def run_one(path, mode, k, max_pairs, frequency, lookahead, suffix_budget):
    automaton = sdfa()
    automaton.load(str(path))
    alphabet = list(range(automaton.num_letters))
    start = time.perf_counter()
    if mode == 'eager':
        worker = sdfa_minimiser()
    elif mode == 'cegar':
        worker = sdfa_cegar_minimiser(verbose=False)
    elif mode in ('cdcl', 'rollback'):
        worker = CDCLProductMinimiser(
            max_pairs=max_pairs,
            check_frequency=frequency,
            lookahead_budget=lookahead,
            suffix_budget=suffix_budget,
            incremental=(mode == 'rollback'),
            verbose=False)
    else:
        raise ValueError(mode)

    sat, candidate = worker.solve('cadical195', k, alphabet, automaton, True, False)
    seconds = time.perf_counter() - start
    candidate_ok = None
    if sat:
        violations, visits = sdfa_cegar_minimiser._verify_product(
            automaton, candidate, limit=1)
        if violations:
            raise AssertionError((mode, k, violations[0]))
        candidate_ok = True
    return dict(status='SAT' if sat else 'UNSAT',
                mode=mode, k=k,
                seconds=round(seconds, 5),
                peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
                verified=candidate_ok,
                stats=getattr(worker, 'stats', None))


def limits(memory_mb):
    def apply():
        if memory_mb:
            size = int(memory_mb * 1024 * 1024)
            resource.setrlimit(resource.RLIMIT_AS, (size, size))
    return apply


def run_subprocess(filename, mode, k, seconds, memory_mb, opts):
    command = [sys.executable, '-u', str(Path(__file__).resolve()),
               '--single', str(filename), '--mode', mode, '--k', str(k),
               '--max-pairs', str(opts.max_pairs),
               '--frequency', str(opts.frequency),
               '--lookahead', str(opts.lookahead),
               '--suffix-budget', str(opts.suffix_budget)]
    try:
        begin = time.perf_counter()
        completed = subprocess.run(
            command, cwd=ROOT, capture_output=True, text=True, timeout=seconds,
            preexec_fn=limits(memory_mb),
            env={**os.environ, 'PYTHONHASHSEED': '0'})
        elapsed = time.perf_counter() - begin
        records = [line[len('BENCH_RESULT '):] for line in completed.stdout.splitlines()
                   if line.startswith('BENCH_RESULT ')]
        if completed.returncode or len(records) != 1:
            return dict(mode=mode, k=k, status='ERROR',
                        seconds=round(elapsed, 5),
                        error=(completed.stderr or completed.stdout)[-1800:])
        result = json.loads(records[0])
        result['subprocess_seconds'] = round(elapsed, 5)
        return result
    except subprocess.TimeoutExpired:
        return dict(mode=mode, k=k, status='TIMEOUT', seconds=seconds)


def execute_suite(args):
    results = []
    with tempfile.TemporaryDirectory(prefix='large3dfa-') as directory:
        for case in args.cases:
            automaton, info = prepare(case)
            path = Path(directory) / (case + '.txt')
            path.write_text(automaton.textual())
            if case.startswith('repo'):
                optimum = KNOWN_REPO_OPT[case]
                target_sizes = [optimum - 1, optimum]
            else:
                # Six-state modular generator; k=5 is expected to be UNSAT,
                # k=6 is guaranteed SAT by the generating machine.
                target_sizes = [5, 6]
            print('INSTANCE', json.dumps(info), flush=True)
            for k in target_sizes:
                if args.only_unsat and k == target_sizes[-1]:
                    continue
                statuses = set()
                for mode in args.modes:
                    row = run_subprocess(path, mode, k, args.seconds,
                                         args.memory_mb, args)
                    row.update(info)
                    results.append(row)
                    if row['status'] in ('SAT', 'UNSAT'):
                        statuses.add(row['status'])
                    print('RESULT', json.dumps(row, sort_keys=True), flush=True)
                if len(statuses) > 1:
                    raise AssertionError('SAT/UNSAT disagreement: ' + repr((case,k,statuses)))
            Path(args.output).write_text(json.dumps(results, indent=2))
    print('BENCHMARK_FINISHED', len(results), 'measurements', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+',
                   default=['repo7', 'repo10', 'sparse600', 'sparse2400'])
    p.add_argument('--modes', nargs='+', choices=DEFAULT_MODES, default=DEFAULT_MODES)
    p.add_argument('--seconds', type=int, default=45, help='timeout per solver in seconds')
    p.add_argument('--memory-mb', type=int, default=3600, help='virtual memory limit per worker')
    p.add_argument('--only-unsat', action='store_true')
    p.add_argument('--output', default='large_3dfa_bench.json')
    p.add_argument('--max-pairs', type=int, default=12000)
    p.add_argument('--frequency', type=int, default=4)
    p.add_argument('--lookahead', type=int, default=32)
    p.add_argument('--suffix-budget', type=int, default=1000)
    p.add_argument('--single', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--mode', choices=DEFAULT_MODES, help=argparse.SUPPRESS)
    p.add_argument('--k', type=int, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.single is not None:
        try:
            result = run_one(args.single,args.mode,args.k,
                             args.max_pairs,args.frequency,args.lookahead,
                             args.suffix_budget)
            print('BENCH_RESULT',json.dumps(result),flush=True)
        except Exception:
            traceback.print_exc()
            raise
    else:
        execute_suite(args)


if __name__ == '__main__':
    main()
