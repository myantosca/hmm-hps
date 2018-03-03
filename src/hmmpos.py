import argparse
#import functools
from itertools import product

"""
Command-line arguments
"""
parser = argparse.ArgumentParser(description='Annotate a sentence with part-of-speech tags.')
parser.add_argument('--training_file', type=str)
parser.add_argument('--test_file', type=str)
parser.add_argument('--n', type=int)
args = parser.parse_args()

TAG_Q0 = "Q0"
TAG_QF = "QF"

# From http://universaldependencies.org/u/pos/
TAGSET = [TAG_Q0, 'ADJ', 'ADP', 'ADV', 'AUX',
          'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM',
          'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ',
          'SYM', 'VERB', 'X', TAG_QF]

"""
Update frequency model with observation and concomitant state annotation.
"""
def add_word_lang_tag(Q, A, B, n, ngram, last_tag, word, lang, tag):
    # Chop off oldest contributor to n-gram cursor if necessary.
    if len(ngram) == n:
        ngram = ngram[1:]
    # No emissions if tag is initial or final state.
    ngram = ngram + ((word, lang),) if (tag != TAG_Q0 and tag != TAG_QF) else ()
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
Convert model of frequencies to probabilities.
"""
def freqs_to_probs(QABn):
    (Q,A,B,n) = QABn
    # Calculate emission probabilities from n-gram counts and state counts.
    for (tag, igram) in B:
        B[(tag, igram)] = B[(tag, igram)] / Q[(tag, len(igram))]
    # Calculate transition probabilities from state transition counts and state counts.
    for (tag1, tag2) in A:
        A[(tag1, tag2)] = A[(tag1, tag2)] / Q[(tag1, 1)]
    # q_count = functools.reduce(lambda x,y: x+y, Q.values()) - Q[(TAG_Q0,1)] - Q[(TAG_QF,i)]
    #     if q_count != 0:
    #         for (tag,j) in Q:
    #             if tag != TAG_Q0 and tag != TAG_QF:
    #                 Q[(tag,j] = Q[tag]/q_count
    Q = [q for (q,i) in Q if i == 1]
    return (Q,A,B,n)

"""
Build model of frequencies from annotated file.
"""
def freqs_from_file(input_file, n):
    # Initialize state counts with Laplace smoothing.
    Q = { (q, i) : 1 for (q,i) in product(TAGSET, [i+1 for i in range(n)]) }
    # Initialize transition frequencies to dictionary populated with all permutations of tag transitions set to 1 (Laplace smoothing). 
    A = { qq : 1 for qq in product(TAGSET, TAGSET) }
    # Initialize emission frequencies to empty dictionary.
    B = {}
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
                (Q, A, B, ngram, last_tag) = add_word_lang_tag(Q, A, B, n, ngram, last_tag, None, None, TAG_QF)
                # Reset ngram cursor
                ngram = ()
                # Reset previous tag to start state.
                last_tag = TAG_Q0
            # Word
            else:
                # Break word annotation by tabs. Grab the first three tokens.
                (word, lang, tag) = tuple(line.split('\t')[0:3])
                # Update model for current annotation.
                (Q, A, B, ngram, last_tag) = add_word_lang_tag(Q, A, B, n, ngram, last_tag, word, lang, tag)
        # Check for implicit end of sentence by EOF (in case annotator forgot final blank line).
        if last_tag != TAG_Q0:
            (Q, A, B, ngram, last_tag) = add_word_lang_tag(Q, A, B, n, ngram, last_tag, None, None, TAG_QF)
    # Return a model of counts.
    return (Q,A,B,n)

"""
Perform a Viterbi decoding given an observation sequence and a HMM.
"""
def viterbi_decode(QABn, O):
    # Return the base case for an empty list of observations.
    if len(O) == 0:
        return [TAG_Q0, TAG_QF]
    (Q,A,B,n) = QABn
    Q = [q for q in Q if q != TAG_QF]
    # Initialization of trellis.
    viterbi = {}
    backptr = {}
    for s in Q:
        b = B[(s,O[0])] if (s,O[0]) in B else 0
        a = A[(TAG_Q0,s)] if (TAG_Q0,s) in A else 0
        #print("s = {}, a = {}, b = {}, O[0] = {}".format(s,a,b,O[0]))
        viterbi[(s,1)] = a * b
        backptr[(s,1)] = TAG_Q0
    t = 2
    for o in O[1:]:
        #print("o = {}".format(o))
        for s in Q:
            amax = 0
            viterbi[(s,t)] = 0
            #backptr[(s,t)] = TAG_Q0
            for r in Q:
                x = A[(r,s)] if (r,s) in A else 0
                a = viterbi[(r,t-1)] * x
                b = B[(s,o)] if (s,o) in B else 0
                #print("r = {}, s = {}, x = {}, a = {}, b = {}".format(r,s,x,a,b))
                viterbi[(s,t)] = max(viterbi[(s,t)], a * b)
                if (a > amax):
                    #print("amax = {}".format(a))
                    backptr[(s,t)] = r
                    amax = a
        t += 1

    viterbi[(TAG_QF, t)] = 0
    for s in Q:
        a = A[(s,TAG_QF)] if (s, TAG_QF) in A else 0
        #print("s = {}, a = {}, viterbi[(s,t-1)] = {}".format(s,a, viterbi[(s,t-1)]))
        if (a * viterbi[(s, t-1)] > viterbi[(TAG_QF, t)]):
            viterbi[(TAG_QF, t)] = a
            backptr[(TAG_QF, t)] = s

    return walk_backpointers(backptr, (TAG_QF, t))

def walk_backpointers(backptr, cursor):
    (tag,t) = cursor
    if (t <= 0):
        return []
    else:
        result = walk_backpointers(backptr, (backptr[(tag,t)], t-1))
        result.append(tag)
        return result
    

def test_model(model, test_file):
    with open(test_file) as fp:
        observations = []
        for line in iter(fp.readline, ''):
            line = line.strip()
            
            # End of sentence
            if len(line) == 0:
                #print(observations)
                tags = viterbi_decode(model, observations)
                for result in zip(observations, tags):
                    #print(result)
                    word = result[0][0][0]
                    lang = result[0][0][1]
                    tag  = result[1]
                    print("{0}\t{1}\t{2}".format(word, lang, tag))
                print("")
                observations = []
            else:
                observations.append((tuple(line.split('\t')[0:2]),))

def train_model(training_file, n):
    return freqs_to_probs(freqs_from_file(training_file, n))
        
"""
main
"""

model = train_model(args.training_file, args.n)

#(Q,A,B,n) = model
# print("***** Q *****")
# print("{0}".format(Q))
# print("***** A *****\n")
# for a in A:
#     print("A[{0}] = {1}".format(a, A[a]))
# print("***** B *****\n")
# for b in B:
#     print("B[{0}] = {1}".format(b, B[b]))

test_model(model, args.test_file)

