# Copyright 2018 Michael Yantosca
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from os import mkdir
import re
import argparse
import subprocess
from subprocess import check_output

"""
Command-line arguments
"""
parser = argparse.ArgumentParser(description='Runs the hmmpos.py part-of-speech tagger and collects stats over the various runs.')
parser.add_argument('--training_file', metavar='TRN', type=str, help='3-column .conll training corpus', required=True)
parser.add_argument('--test_file', metavar='TST', type=str, help='2-column .conll file (w/o POS tags) test corpus', required=True)
parser.add_argument('--output_dir', metavar='OUT', type=str, help='output directory for results', default='.')

args = parser.parse_args()

mkdir(args.output_dir)

word_count = int(check_output("egrep -v '^$' {} | wc -l".format(args.test_file), shell=True))

gold_pos={}
gold_neg={}

# From http://universaldependencies.org/u/pos/
TAGSET = ['ADJ', 'ADP', 'ADV', 'AUX',
          'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
          'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ',
          'SYM', 'VERB', 'X', 'UNK']

for tag in TAGSET:
    gold_pos[tag] = int(check_output("egrep '{}$' {} | wc -l".format(tag, args.test_file), shell=True))
    gold_neg[tag] = word_count - gold_pos[tag]

sts_file = args.output_dir + "/results.stats"

with open(sts_file, mode='w') as fp:
    fp.write("trial,k,n,backoff,heuristic,word_count,errors,")
    for tag in TAGSET:
        fp.write("FP({}),FN({}),TP({}),TN({}),P({}),R({}),A({}),F1({})".format(tag, tag, tag, tag, tag, tag, tag, tag))
    fp.write("\n")
    # Repeated trials
    for trial in range(5):
        for k in [1, 0.1, 0.01, 0.001]:
            for n in [i+1 for i in range(4)]:
                for heuristic in ['no_heuristic_fallback', 'heuristic_fallback', 'heuristic_preseed']:
                    for backoff in ['backoff', 'weights']:
                        heur_abbr = re.sub('heuristic_', '', heuristic)
                        file_prefix = "{}/train-dev-{}-{}-{}-{}-{}".format(args.output_dir, trial, n, k, backoff, heur_abbr)

                        out_file = file_prefix + ".conll"
                        err_file = file_prefix + ".err"
                        out_line = "{},{},{},{},{},".format(n, k, backoff, heur_abbr, trial)
                        command="python3 hmmpos.py --training_file {} --test_file {} --n {} --k {} --ngram_{} --{} 1> {} 2> {}".format(args.training_file, args.test_file, n, k, backoff, heuristic, out_file, err_file)
                        print(command)
                        check_output(command, shell=True)
                        errors = int(check_output("diff {} {} | egrep '<' | wc -l".format(args.test_file, out_file), shell=True))
                        fp.write("{},{},{},{},{},{},{},".format(trial, k, n, backoff, heuristic, word_count, errors))
                        for tag in TAGSET:
                            false_pos = int(check_output("diff {} {} | egrep '<' | egrep '{}$' | wc -l".format(args.test_file, out_file, tag), shell=True))
                            false_neg = int(check_output("diff {} {} | egrep '<' | egrep '{}$' | wc -l".format(args.test_file, out_file, tag), shell=True))
                            true_pos  = gold_pos[tag] - false_pos
                            true_neg  = gold_neg[tag] - false_neg
                            precision = 0 if true_pos + false_pos == 0 else float(true_pos) / float(true_pos + false_pos)
                            recall    = 0 if true_pos + false_neg == 0 else float(true_pos) / float(true_pos + false_neg)
                            accuracy  = 0 if true_pos + false_pos + true_neg + false_neg == 0 else float(true_pos + true_neg) / float(true_pos + false_pos + true_neg + false_neg) 
                            f1measure = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                            fp.write("{},{},{},{},{},{},{},{}".format(false_pos, false_neg, true_pos, true_neg, precision, recall, accuracy, f1measure))
                        fp.write("\n")
