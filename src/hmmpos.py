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

# Submitted under the handle 'ghostant' to COSC 6336 - Part-of-Speech Tagging with HMM CodaLab Competition

import argparse
from itertools import product
import re
from math import log,pow,e
import sys

"""
Command-line arguments
"""
parser = argparse.ArgumentParser(description='Annotates a series of sentences with part-of-speech tags.')
parser.add_argument('--training_file', metavar='TRN', type=str, help='3-column .conll training corpus', required=True)
parser.add_argument('--test_file', metavar='TST', type=str, help='2-column .conll file (w/o POS tags) test corpus', required=True)
parser.add_argument('--n', type=int, help='highest order of n-gram to include in features', default=1)
parser.add_argument('--k', type=float, help='add-k smoothing constant', default=0.01)
ngram_backoff = parser.add_mutually_exclusive_group()
ngram_backoff.add_argument('--ngram_backoff', action='store_true', help='backs off emission probs on n-grams until 1-gram')
ngram_backoff.add_argument('--ngram_weights', action='store_false', help='composites emission probs on n-grams with weights [iteratively 0.75 of remainder until 1-gram]')
heuristic_fallback=parser.add_mutually_exclusive_group()
heuristic_fallback.add_argument('--heuristic_fallback', action='store_true', help='heuristic check during decoding for unknown tokens')
heuristic_fallback.add_argument('--no_heuristic_fallback', action='store_false')
heuristic_preseed=parser.add_mutually_exclusive_group()
heuristic_preseed.add_argument('--heuristic_preseed', action='store_true', help='heuristic preseed of emission probs for unknown tokens')
heuristic_preseed.add_argument('--no_heuristic_preseed', action='store_false')

args = parser.parse_args()

if args.heuristic_preseed and args.heuristic_fallback:
    args.error('--heuristic_preseed and --heuristic_fallback are mutually exclusive. Please choose one or the other.')

TAG_Q0 = "Q0"
TAG_QF = "QF"
WORD_UNK = "<UNK>"
LANG_UNK = "eng&spa"
MIN_PROB = -sys.float_info.max
MAX_PROB = log(1.0)

RE_SOME_WORD = re.compile('[A-ZÁÉÍÓÚÑa-záéíóúñ0-9]')
RE_CAP       = re.compile('^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑa-záéíóúñ_0-9]+$')
RE_ALL_PUNCT = re.compile('^["\'.?,;:!¿¡]+$')

# From http://universaldependencies.org/u/pos/
TAGSET = [TAG_Q0, 'ADJ', 'ADP', 'ADV', 'AUX',
          'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
          'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ',
          'SYM', 'VERB', 'X', 'UNK', TAG_QF]

CLOSED_TAGS = ['ADP', 'AUX', 'CCONJ', 'DET', 'NUM', 'PART', 'PRON', 'SCONJ']

"""
Updates frequency model with observation and concomitant state annotation.

Q = state counts
A = transition counts
B = observation counts
n = highest order of n-gram observation history
preseed = flag whether to heuristically pre-seed emission probabilities for unknown tokens
ngram = cursor keeping observation history up to nth order
last_tag = last POS tag seen
word = current word to add to the observation
lang = language ID for current word
tag = current POS tag
"""
def add_word_lang_tag(Q, A, B, n, preseed, ngram, last_tag, word, lang, tag):
    # Chop off oldest contributor to n-gram cursor if necessary.
    if len(ngram) == n:
        ngram = ngram[1:]
    # Calculate dependent features.
    if not word is None:
        some_word = (not RE_SOME_WORD.match(word) is None) if preseed else None
        cap = (not RE_CAP.match(word) is None) if preseed else None
        all_punct = (not RE_ALL_PUNCT.match(word) is None) if preseed else None
    # No emissions if tag is initial or final state.
    ngram = ngram + ((word, lang, some_word, cap, all_punct),) if (tag != TAG_Q0 and tag != TAG_QF) else ()
    # Increment count of last_tag -> tag in transition probability matrix.
    A[(last_tag, tag)] = 1 if not ((last_tag, tag) in A) else A[(last_tag, tag)] + 1
    # update n-gram counts for all orders of n-gram
    for i in range(len(ngram)):
        igram = ngram[i:]
        isize = len(igram)
        # Increment population count of tag respective the observation igram size.
        Q[(tag, isize)] = 1 if (not ((tag, isize) in Q)) else Q[(tag, isize)] + 1
        # Add count to emission probability matrix.
        B[(tag, igram)] = 1 if (not ((tag, igram) in B)) else B[(tag, igram)] + 1
    return (Q, A, B, ngram, tag)

"""
Converts a model of frequencies to one of probabilities.

NB: Some of the heuristic shortcuts that are taken make it so
that the model cannot be considered to be a true probability
distribution. The reader is referred to the work by
Brants, et al. (2007) on stupid backoff. Some attempts
are made to preserve the probability mass, but a more
thorough refactoring is needed.

QABnk = model of frequencies
Q = state counts
A = transition counts
B = observation counts
n = highest order n-gram
k = add-k smoothing constant
"""
def freqs_to_probs(QABnk):
    (Q,A,B,n,k) = QABnk
    # Determine the size of the 1-gram vocabulary.
    V = len([(t,g) for (t,g) in B.keys() if len(g) == 1])
    # Calculate emission probabilities from n-gram counts and state counts.
    for (tag, igram) in [(tag, igram) for (tag, igram) in B.keys() if (tag != TAG_Q0 and tag != TAG_QF)]:
        isize = len(igram)
        B[(tag, igram)] = log((B[(tag, igram)] + k) / (Q[(tag, isize)] + k*V))
    # Calculate transition probabilities from state transition counts and state counts.
    for (tag1, tag2) in A.keys():
        A[(tag1, tag2)] = log(A[(tag1, tag2)] / Q[(tag1, 1)])
    for (tag1, tag2) in A.keys():
        if tag1 == TAG_QF or tag2 == TAG_Q0:
            A[(tag1, tag2)] = MIN_PROB
    Q = [q for (q,i) in Q if i == 1]
    # TODO: Adjust probabilities for unknown words if preseed is in effect.
    for (tag, igram) in [(tag, igram) for (tag, igram) in B.keys() if igram[len(igram)-1][0] == WORD_UNK and not igram[len(igram)-1][2] is None]:
        index = len(igram)-1
        if tag == TAG_Q0 or tag == TAG_QF:
            B[(tag,igram)] = MIN_PROB
        elif tag in CLOSED_TAGS:
            B[(tag,igram)] = MIN_PROB
        elif igram[index][3]:
            B[(tag,igram)] = MIN_PROB if tag != 'PROPN' or igram[index][4] else MAX_PROB
        elif tag == 'PROPN':
            B[(tag,igram)] = MIN_PROB
        elif igram[index][2]:
            B[(tag,igram)] = MIN_PROB if tag == 'PUNCT' or tag == 'SYM' or tag == 'INTJ' or tag == 'UNK' or tag == 'X' or igram[index][4] else log(pow(e, B[(tag,igram)]) * 1.125)
        elif igram[index][4]:
            B[(tag,igram)] = MIN_PROB if tag != 'PUNCT' else MAX_PROB
    return (Q,A,B,n,k)

"""
Generates the initial emission probability matrix with entries for unknown 1-grams.

preseed = flag whether to heuristically pre-seed emission probabilities for unknown tokens
"""
def generate_emission_matrix(preseed):
    if (not preseed):
        return { (q, ((WORD_UNK, LANG_UNK, None, None, None),)) : 0 for q in TAGSET }
    else:
        # key = (word, lang, word =~ /\w+/, word =~ /^[A-ZA-ZÁÉÍÓÚÑ]/, word =~ /^["\'.?,;:!¿¡]+$/)
        return { (q, ((WORD_UNK, LANG_UNK, SOME_WORD, CAP, ALL_PUNCT),)) : 0 for (q, SOME_WORD, CAP, ALL_PUNCT) in product(TAGSET, [True, False], [True, False], [True, False]) }

"""
Builds a model of frequencies from an annotated training file.

input_file = path to the 3-column .conll file to act as the training corpus
n = highest n-gram order to use in features
k = add-k smoothing constant
preseed = flag whether to heuristically pre-seed emission probabilities for unknown tokens
"""
def freqs_from_file(input_file, n, k, preseed):
    # Initialize state counts with Laplace smoothing.
    Q = { (q, i) : k * len(TAGSET) for (q,i) in product(TAGSET, [i+1 for i in range(n)]) }
    # Initialize transition frequencies to dictionary populated with all permutations of tag transitions set to 1 (Laplace smoothing).
    A = { qq : k for qq in product(TAGSET, TAGSET) }
    # Initialize emission frequencies with Laplace smoothing for unknown words.
    B = generate_emission_matrix(preseed)
    # Initialize ngram cursor to empty tuple.
    ngram = ()
    # Initialize last tag to start state.
    last_tag = TAG_Q0
    # https://docs.python.org/3/library/functions.html#open
    with open(input_file) as fp:
        for line in iter(fp.readline, ''):
            line = line.strip()
            # End of sentence
            if len(line) == 0:
                # Count matching start state
                Q[(TAG_Q0, 1)] += 1
                # Update model for end of sentence
                (Q, A, B, ngram, last_tag) = add_word_lang_tag(Q, A, B, n, preseed, ngram, last_tag, None, None, TAG_QF)
                # Reset ngram cursor
                ngram = ()
                # Reset previous tag to start state.
                last_tag = TAG_Q0
            # Word
            else:
                # Break word annotation by tabs. Grab the first three tokens.
                (word, lang, tag) = tuple(line.split('\t')[0:3])
                # Update model for current annotation.
                (Q, A, B, ngram, last_tag) = add_word_lang_tag(Q, A, B, n, preseed, ngram, last_tag, word, lang, tag)
        # Check for implicit end of sentence by EOF (in case annotator forgot final blank line).
        if last_tag != TAG_Q0:
            (Q, A, B, ngram, last_tag) = add_word_lang_tag(Q, A, B, n, preseed, ngram, last_tag, None, None, TAG_QF)
    # Return a model of counts.
    return (Q,A,B,n,k)


"""
Performs a final fallback to emission probabilities for unknown tokens during Viterbi decoding.
If the heuristic_fallback flag is set, this involves expensive regex matching.
For best results, it is recommended to set the --heuristic_preseed flag at the command line,
which roughly translates the regex matches done here into pre-seeded emission probabilities
so that the feature derivation is implicitly encoded into the emission probability matrix.

The heuristic checks primarily cover handling of proper nouns, punctuation, and unknown closed tags.
NB: the return of MIN_PROB does not redistribute the probability mass, and so it cannot
be considered a true probability distribution, similar to stupid backoff by Brants, et al. (2007).

B = emission probability matrix
s = POS tag
o = observation/feature vector
n = order of feature vector
heuristic_fallback = flag whether to do heuristic checks on final fallback
"""
def observation_fallback(B,s,o,n,heuristic_fallback):
    if heuristic_fallback:
        if re.match('^[A-ZÁÉÍÓÚÑ][a-záéíóúñ]*$', o[n-1][0]) and s != 'PROPN':
            return MIN_PROB
        elif re.match('^[^A-zÁÉÍÓÚáéíóúñ0-9]+$', o[n-1][0]) and s != 'SYM' and s != 'PUNCT':
            return MIN_PROB
        elif re.match('^[A-ZÁÉÍÓÚÑa-záéíóúñ]+$', o[n-1][0]) and (s == 'SYM' or s == 'PUNCT'):
            return MIN_PROB
        elif re.match('^[^A-ZÁÉÍÓÚÑ]+$', o[n-1][0]) and s == 'PROPN':
            return MIN_PROB
        elif s in CLOSED_TAGS:
            return MIN_PROB
        elif re.match('^["\'.?,;:!¿¡]+$', o[n-1][0]) and s != 'PUNCT':
            return MIN_PROB
    return B[(s, ((WORD_UNK,LANG_UNK) + o[n-1][2:],))]


"""
Retrieves emission probabilities for a feature vector according to a composition model.
The backoff starts with the highest order ngram and checks for existence in the
emission probability matrix. If the n-gram is a valid key, it uses the found
probability. If not, it does a fallback to the corresponding unknown 1-gram.
The resultant value is multiplied by a weight of 0.75 and then added to
successive weighted probabilities for each order of n-gram until the 1-gram.
Each successive n-gram takes 0.75 of the remaining weight until the 1-gram
takes the final remaining 0.25.

B = emission probability matrix
s = POS tag
ngram = feature vector
t = deprecated parameter indicating the observation index in the overall sequence that really ought to be removed
heuristic_fallback = flag whether to do heuristic checks on final fallback
"""
def observation_composite(B,s,ngram,t,heuristic_fallback):
    weight = 1.0
    result = 0.0
    n = len(ngram)
    for i in range(n):
        igram = ngram[i:]
        isize = len(igram)
        if isize <= t:
            if (i < n - 1):
                weight = weight - weight * 0.25
            result += weight * pow(e, B[(s,igram)] if (s,igram) in B else observation_fallback(B,s,igram,isize,heuristic_fallback))
    return log(result) if (result > 0) else MIN_PROB

"""
Retrieves emission probabilities for a feature vector according to a backoff model.
The backoff starts with the highest order ngram and checks for existence in the
emission probability matrix until finally arriving at the 1-gram.
If the 1-gram is an invalid key, it falls back to an unknown token.

B = emission probability matrix
s = POS tag
ngram = feature vector
heuristic_fallback = flag whether to do heuristic checks on final fallback
"""
def observation_backoff(B,s,ngram,heuristic_fallback):
    if len(ngram) == 0:
        return MIN_PROB
    elif len(ngram) == 1:
        return B[(s,ngram)] if (s,ngram) in B else observation_fallback(B,s,ngram,1,heuristic_fallback)
    else:
        return B[(s,ngram)] if (s,ngram) in B else observation_backoff(B,s,ngram[1:],heuristic_fallback)

"""
Performs a Viterbi decoding given an observation sequence and a HMM.
The algorithm is an implementation of the pseudocode given in
Fig 9.11 in Chapter 9 of Speech and Language Processing 3rd ed.
by Jurasky and Martin.

QABnk = HMM to use in decoding
O = observed sentence (with language ID)
ngram_backoff = flag whether to do backoff or weighted composition of emission probabilities
heuristic_fallback = flag whether to do heuristic checks during decoding
"""
def viterbi_decode(QABnk, O, ngram_backoff, heuristic_fallback):
    # Return the base case for an empty list of observations.
    if len(O) == 0:
        return [TAG_Q0, TAG_QF]
    (Q,A,B,n,k) = QABnk
    Q = [q for q in Q if q != TAG_QF and q != TAG_Q0]
    # Initialization of trellis.
    viterbi = {}
    backptr = {}
    for s in Q:
        b = B[(s,O[0])] if (s,O[0]) in B else observation_fallback(B,s,O[0],1,heuristic_fallback)
        a = A[(TAG_Q0,s)] if (TAG_Q0,s) in A else MIN_PROB
        viterbi[(s,1)] = a + b
        backptr[(s,1)] = TAG_Q0
    t = 2
    ngram = O[0]
    for o in O[1:]:
        if (len(ngram) == n):
            ngram = ngram[1:]
        ngram += o
        for s in Q:
            amax = MIN_PROB
            viterbi[(s,t)] = MIN_PROB
            for r in Q:
                x = A[(r,s)] if (r,s) in A else MIN_PROB
                a = viterbi[(r,t-1)] + x
                if ngram_backoff:
                    b = observation_backoff(B, s, ngram, heuristic_fallback)
                else:
                    b = observation_composite(B, s, ngram, t, heuristic_fallback)
                viterbi[(s,t)] = max(viterbi[(s,t)], a + b)
                if (a > amax):
                    backptr[(s,t)] = r
                    amax = a
        t += 1

    viterbi[(TAG_QF, t)] = MIN_PROB
    for s in Q:
        a = A[(s,TAG_QF)] if (s, TAG_QF) in A else MIN_PROB
        if (a + viterbi[(s, t-1)] > viterbi[(TAG_QF, t)]):
            viterbi[(TAG_QF, t)] = a
            backptr[(TAG_QF, t)] = s

    return walk_backpointers(backptr, (TAG_QF, t))

"""
Walks the back-pointers given by a Viterbi decoding to find
the most probable sequence of hidden states.

backptr = the dictionary of back-pointers
cursor = the key from which to start the traversal
"""
def walk_backpointers(backptr, cursor):
    (tag,t) = cursor
    if (t <= 0):
        return []
    else:
        result = walk_backpointers(backptr, (backptr[(tag,t)], t-1))
        result.append(tag)
        return result

"""
Tests a HMM generated by train_model against a given test file.

model = the generated model
test_file = path to the 2-column .conll file to be tested and annotated
ngram_backoff = flag whether to do backoff or weighted composition of emission probabilities
heuristic_fallback = flag whether to do heuristic checks during decoding
preseed = flag whether to presume heuristic pre-seeding of emission probabilities for unknown tokens
"""
def test_model(model, test_file, ngram_backoff, heuristic_fallback, preseed):
    with open(test_file) as fp:
        observations = []
        for line in iter(fp.readline, ''):
            line = line.strip()
            if len(line) == 0:
                # End of sentence
                tags = viterbi_decode(model, observations, ngram_backoff, heuristic_fallback)
                # Combine given observations and resultant tags and output annotated sentence to stdout.
                for result in zip(observations, tags):
                    word = result[0][0][0]
                    lang = result[0][0][1]
                    tag  = result[1]
                    print("{0}\t{1}\t{2}".format(word, lang, tag))
                # Add end of sentence marker.
                print("")
                observations = []
            else:
                # Build feature vector for a word and language id.
                ngram = (tuple(line.split('\t')[0:2]))
                ngram += ((not RE_SOME_WORD.match(ngram[0]) is None) if preseed else None,)
                ngram += ((not RE_CAP.match(ngram[0]) is None) if preseed else None,)
                ngram += ((not RE_ALL_PUNCT.match(ngram[0]) is None) if preseed else None,)
                # Add feature vector to current sentence.
                observations.append((ngram,))

"""
Trains a HMM given a 3-column .conll training corpus.

training_file = path to the 3-column .conll file to act as the training corpus
n = highest n-gram order to use in features
k = add-k smoothing constant
preseed = flag whether to heuristically pre-seed emission probabilities for unknown tokens
"""
def train_model(training_file, n, k, preseed):
    return freqs_to_probs(freqs_from_file(training_file, n, k, preseed))

"""
main
"""

model = train_model(args.training_file, args.n, args.k, args.heuristic_preseed)

# (Q,A,B,n,k) = model
# print("***** Q *****")
# print("{0}".format(Q))
# print("***** A *****\n")
# for a in A:
#     print("A[{0}] = {1}".format(a, A[a]))
# print("***** B *****\n")
# for b in B:
#     print("B[{0}] = {1}".format(b, B[b]))

test_model(model, args.test_file, args.ngram_backoff, args.heuristic_fallback, args.heuristic_preseed)
