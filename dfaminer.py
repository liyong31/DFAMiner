"""
DFA Miner module for inferring minimal Deterministic Finite Automata from samples.

This module provides functionality to mine DFAs consistent with positive and negative
samples using SAT-based techniques for minimization. It supports multiple input formats
(JSON, Python, and ABBADINGO formats) and can output results in textual or DOT format.
"""

import sdfa as SDFA
import strunion
import minimiser

from functools import cmp_to_key
import sys
import time


class dfa_miner:
    """
    The class for mining minimal DFAs from positive and negative word samples.
    
    This class handles reading sample data, constructing DFAs (both regular and
    three-valued), and minimizing them using SAT solvers. It supports various input
    formats and can verify the resulting DFA against the input samples.
    
    Attributes:
        OUTPUT_FORMAT (list): Supported output formats ('dot', 'textual').
        OUTPUT_FORMAT_DEFAULT (str): Default output format ('textual').
        positive_samples (list): List of accepted word samples.
        negative_samples (list): List of rejected word samples.
        num_samples (int): Total number of samples.
        num_letters (int): Size of the alphabet.
        has_emptysample (bool): Whether the empty word is a sample.
        accept_empty (bool): Whether the empty word should be accepted.
        alphabet (list/range): The alphabet symbols.
        args (argparse.Namespace): Parsed command-line arguments.
    """
    OUTPUT_FORMAT = ['dot', 'textual']
    OUTPUT_FORMAT_DEFAULT = 'textual'

    def __init__(self):
        """Initialize a DFA miner instance with empty samples and default settings."""
        self.positive_samples = []
        self.negative_samples = []
        self.num_samples = 0
        self.num_letters = 0
        self.has_emptysample= False
        self.accept_empty = False
        self.args = None

    def verify_conjecture_dfa(self, dfa):
        """Verify that the inferred DFA correctly classifies all input samples.
        
        Tests the DFA against all positive and negative samples to ensure correctness.
        Exits with error code -1 if any misclassification is found.
        
        Args:
            dfa: The DFA to verify.
            
        Raises:
            SystemExit: If the DFA misclassifies any sample.
        """
        # we verify by enumerating all possible words
        init = None
        for i in dfa.init_states:
            init = i
            break

        for p in self.positive_samples:
            # print("word: = " + str(p))
            res = dfa.run(init, p)
            # print("label: = " + str( res == strunion.word_type.ACCEPT))
            if res != strunion.word_type.ACCEPT:
                print("ERROR classification: ", p, " Sample value: ", True)
                sys.exit(-1)
        for p in self.negative_samples:
            # print("word: = " + str(p))
            res = dfa.run(init, p)
            # print("label: = " + str( res == strunion.word_type.ACCEPT))
            if res == strunion.word_type.ACCEPT:
                print("ERROR classification: ", p, " Sample value: ", False)
                sys.exit(-1)
        
        print("DFA verification passed")

    def get_word(self, line_brk):
        """Parse a word from a line in ABBADINGO format.
        
        Args:
            line_brk (list): A line split into tokens from ABBADINGO format.
                           Format: [mq, num, letter1, letter2, ...]
                           where mq is membership query (1=accept, 0=reject)
                           and num is the word length.
            
        Returns:
            tuple: A tuple (mq, word) where mq indicates acceptance and word is
                   a list of integer-encoded letters.
        """
        # print("break lines: ", list(line_brk), " #", len(line_brk))
        mq = int(line_brk[0])
        num = int(line_brk[1])
        w = [int(i) for i in line_brk[2:(2+ num)]]
        return (mq, w)

    def read_samples(self, file_name):
        """Read samples from a file in JSON, Python, or ABBADINGO format.
        
        Automatically detects the file format based on file extension:
        - .json: JSON format with 'alphabet', 'accepting', 'rejecting' fields
        - .py: Python format with 'positive_samples' and 'negative_samples' lists
        - other: ABBADINGO format
        
        Args:
            file_name (str): Path to the sample file.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        import os
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"File not found: {file_name}")
        
        extension = file_name.split('.')[-1]
        if extension == "json":
            self.read_samples_json(file_name)
        elif extension == "py":
            self.read_samples_py(file_name)
        else:
            self.read_samples_abbadingo(file_name)
    
    def read_samples_json(self, file_name):
        """Read samples from a JSON file.
        
        Expected JSON format:
        {
            "alphabet": [...],
            "accepting": [[word1], [word2], ...],
            "rejecting": [[word1], [word2], ...]
        }
        
        Args:
            file_name (str): Path to the JSON sample file.
        """
        with open(file_name, "r") as f:
            import json
            samples = json.load(f)

        pos_samples = samples['accepting']
        neg_samples = samples['rejecting']
        if 'alphabet' in samples:
            alphabet = samples['alphabet']
        else:
            # reconstruct the used alphabet
            alphabet = sorted(set(letter for all_samples in pos_samples + neg_samples for letter in all_samples))

        # convert the input data to the internal format
        self.convert_input(alphabet, pos_samples, neg_samples)

    def read_samples_py(self, file_name):
        """Read samples from a Python file.
        
        Expected Python file format:
        positive_samples = [[word1], [word2], ...]
        negative_samples = [[word1], [word2], ...]
        
        The alphabet is automatically inferred from the samples.
        
        Args:
            file_name (str): Path to the Python sample file.
            
        Raises:
            ValueError: If the Python file cannot be executed.
        """
        # read the file content
        with open(file_name, 'r') as f:
            python_content = f.read()
        
        # create a namespace to execute the file content
        namespace = {}
        try:
            exec(python_content, namespace)
        except Exception as e:
            raise ValueError(f"Error executing file {file_name}: {e}")
        
        # extract positive_samples and negative_samples
        if 'positive_samples' in namespace:
            pos_samples = namespace['positive_samples']
        else:
            pos_samples = []
        if 'negative_samples' in namespace:
            neg_samples = namespace['negative_samples']
        else:
            neg_samples = []
        # reconstruct the used alphabet
        alphabet = sorted(set(letter for all_samples in pos_samples + neg_samples for letter in all_samples))

        # convert the input data to the internal format
        self.convert_input(alphabet, pos_samples, neg_samples)

    def read_samples_abbadingo(self, file_name):
        """Read samples from an ABBADINGO format file.
        
        ABBADINGO format:
        Line 1: num_samples num_letters
        Following lines: mq num letter1 letter2 ... letternum
        where mq=1 for accepting, mq=0 for rejecting, -1 for comments.
        
        Args:
            file_name (str): Path to the ABBADINGO format file.
        """
        with open(file_name, "r") as f:
            # read line by line
            line_idx = 0
            for line in f:
                line_brk = line.split()
                if len(line_brk) < 2:
                    continue
                elif line_brk[0] == "-1":
                    continue
                elif line_idx <= 0:
                    self.num_samples = int(line_brk[0])
                    self.num_letters = int(line_brk[1])
                    self.alphabet = range(self.num_letters)
                else:
                    mq, w = self.get_word(line_brk)
                    # special case for empty sample
                    if len(w) == 0:
                        self.has_emptysample = True
                        self.accept_empty = (mq == 1)
                        continue
                    if mq == 1:
                        self.positive_samples.append(w)
                    elif mq == 0:
                        self.negative_samples.append(w)
                line_idx += 1

        # now sort them in place
        self.positive_samples.sort(key=cmp_to_key(strunion.dfa_builder.LEXICOGRAPHIC_ORDER))
        self.negative_samples.sort(key=cmp_to_key(strunion.dfa_builder.LEXICOGRAPHIC_ORDER))

    def convert_input(self, alphabet, pos_samples, neg_samples):
        """Convert input samples to internal representation.
        
        Converts alphabet symbols to integer indices and normalizes sample format.
        Handles the special case of the empty word.
        
        Args:
            alphabet (list): The alphabet symbols.
            pos_samples (list): Accepting word samples.
            neg_samples (list): Rejecting word samples.
        """
        self.alphabet = alphabet
        self.num_letters = len(alphabet)
        self.num_samples = len(pos_samples) + len(neg_samples)

        # check whether the empty word is one of the samples
        if ([] in pos_samples) or ([] in neg_samples):
            self.has_emptysample = True
            self.accept_empty = ([] in pos_samples)

        # letters in the alphabet are represented internally as natural numbers, so convert them accordingly
        self.positive_samples = [[(lambda x: alphabet.index(x))(letter) for letter in sample] for sample in pos_samples]
        self.negative_samples = [[(lambda x: alphabet.index(x))(letter) for letter in sample] for sample in neg_samples]
       
        # now sort the samples in place
        self.positive_samples.sort(key=cmp_to_key(strunion.dfa_builder.LEXICOGRAPHIC_ORDER))
        self.negative_samples.sort(key=cmp_to_key(strunion.dfa_builder.LEXICOGRAPHIC_ORDER))

    def infer_min_dfa(self):
        """Infer a minimal DFA consistent with the samples.
        
        Constructs an intermediate DFA (either three-valued or union of two DFAs)
        from the samples, then minimizes it using a SAT solver.
        
        Returns:
            A minimized DFA that accepts all positive samples and rejects all
            negative samples.
        """
        sdfa = None
        if self.args.sdfa:
            # create the SDFA
            samples = [ (sample, True) for sample in self.positive_samples]
            samples.extend([(sample, False) for sample in self.negative_samples])
            WORD_ORDER = lambda s1, s2: strunion.dfa_builder.LEXICOGRAPHIC_ORDER(s1[0], s2[0])
            # now we sort them in place
            samples.sort(key=cmp_to_key(WORD_ORDER))
            builder = strunion.dfa_builder()
            for sample in samples:
                builder.add(sample[0], strunion.word_type.ACCEPT 
                            if sample[1] else strunion.word_type.REJECT)
            sdfa = strunion.dfa_builder.build(builder, self.num_letters)
        else:
            # create two regular DFAs
            builder = strunion.dfa_builder()
            # positive examples
            for sample in self.positive_samples:
                builder.add(sample, strunion.word_type.ACCEPT)
            pos_dfa = strunion.dfa_builder.build(builder, self.num_letters)
            print("# of states in positive DFA: ", pos_dfa.num_states)
            
            builder = strunion.dfa_builder()
            # positive examples
            for sample in self.negative_samples:
                builder.add(sample, strunion.word_type.ACCEPT)
            neg_dfa = strunion.dfa_builder.build(builder, self.num_letters)
            print("# of states in negative DFA: ", neg_dfa.num_states)
            # print(neg_dfa.dot())
            
            sdfa = SDFA.sdfa.combine(pos_dfa, neg_dfa, self.num_letters)
            # print(sdfa.dot())
            
        # handle possible empty sample
        if self.has_emptysample:
            func = lambda x: sdfa.add_final_state(x) if self.accept_empty else sdfa.add_reject_state(x)
            # initial states would not have incoming edges
            for init in sdfa.init_states:
                func(init)

        # output intermediate file, if needed
        if args.intermediate:
            print("Output intermediate 3DFA to " + args.intermediate)
            self.output_result(sdfa, args.intermediate, args.output_format)
        
        # now minimise
        min = minimiser.sdfa_minimiser()
        result_dfa = min.minimise(input_sdfa=sdfa, sat=self.args.solver
            , lbound=self.args.lower, ubound=self.args.upper, nobfs=self.args.nobfs, safety=self.args.safety)
        
        return result_dfa

    def output_result(self, dfa, output_path, output_format):
        """Write the DFA to a file in the specified format.
        
        Args:
            dfa: The DFA to output.
            output_path (str): Path to the output file.
            output_format (str): Output format ('textual' or 'dot').
        """
        if output_path.split('.')[-1] == 'dot' and output_format != 'dot':
            print ("Warning: inconsistent DOT output file format. File extension: .dot; actual content: textual")
        with open(output_path, "w") as file:
            if output_format == 'textual': 
                file.write(dfa.textual())
            elif 'dot':
                file.write(dfa.dot(self.alphabet))
            else:
                print(f"Output format '{output_format}' not supported")

# sorted(data, key=cmp_to_key(custom_comparator))
if __name__ == '__main__':
    """
    Main entry point for the DFA Miner tool.
    
    Parses command-line arguments and runs the DFA mining process:
    1. Reads sample data from the specified input file
    2. Infers a minimal DFA consistent with the samples
    3. Optionally verifies the DFA
    4. Outputs the result to the specified file
    """
    # instantiate the command line options parser
    import argparse
    parser = argparse.ArgumentParser(description='Mining a minimal DFA consistent with samples')
    parser.add_argument('--file', metavar='path', type=str, required=True,
                        help='path to input sample file')
    parser.add_argument('--out', metavar='path', type=str, required=True,
                        help='path to output DFA')
    parser.add_argument('--output-format', type=str.lower, required=False,
                        choices=dfa_miner.OUTPUT_FORMAT, default=dfa_miner.OUTPUT_FORMAT_DEFAULT,
                        help='the format for the output (default: %(default)s)')
    parser.add_argument('--intermediate', metavar='path', type=str, required=False,
                        default=None,
                        help='path to output the intermediate 3DFA')
    parser.add_argument('--solver', type=str.lower, required=False,
                        choices=minimiser.solver_choices, default="cadical153",
                        help='choose the SAT solver (default: %(default)s)')
    parser.add_argument('--lower', type=int, required=False,
                        default=1,
                        help='the lower bound for the DFA (default: %(default)s)')
    parser.add_argument('--upper', type=int, required=False,
                        default=sys.maxsize,
                        help='the upper bound for the DFA')
    parser.add_argument('--3dfa', action="store_true", required=False,
                        default=False, dest='sdfa',
                        help='use three valued DFA for inference')
    parser.add_argument('--nobfs', action="store_true", required=False,
                        default=False,
                        help='disable the constraints for BFS tree')
    parser.add_argument('--safety', action="store_true", required=False,
                        default=False,
                        help='construct safety DFA for solving parity games')
    parser.add_argument('--verify', action="store_true", required=False,
                        default=False,
                        help='verify resultant DFA')
    args = parser.parse_args()
    
    # start the execution of dfaminer
    start_time = time.time()
    miner = dfa_miner()
    miner.read_samples(args.file)
    print("Input alphabet size: ", miner.num_letters)
    miner.args = args
    result_dfa = miner.infer_min_dfa()
    
    if args.verify:
        miner.verify_conjecture_dfa(result_dfa)
        
    print("Output to " + args.out)
    miner.output_result(result_dfa, args.out, args.output_format)
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 4)
    print(f"Elapsed time in miner: {elapsed_time} secs")