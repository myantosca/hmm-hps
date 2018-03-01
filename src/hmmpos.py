import argparse
import functools

"""
Command-line arguments
"""
parser = argparse.ArgumentParser(description='Annotate a sentence with part-of-speech tags.')
parser.add_argument('--input_file', type=str)
parser.add_argument('--n', type=int)
args = parser.parse_args()

TAG_Q0 = "Q0"
TAG_QF = "QF"

"""
Update frequency model with observation and concomitant state annotation.
"""
def add_word_lang_tag(Q, A, B, n, ngram, last_tag, word, lang, tag):
    # Chop off oldest contributor to n-gram cursor if necessary.
    if len(ngram) == n:
        ngram = ngram[1:]
    # No emissions if tag is initial or final state.
    ngram = ngram + ((word, lang),) if (tag != TAG_Q0 and tag != TAG_QF) else ()
    # Increment population count of tag.
    Q[tag] = 1 if (not (tag in Q)) else Q[tag] + 1
    # Increment count of last_tag -> tag in transition probability matrix.
    A[(last_tag, tag)] = 1 if not ((last_tag, tag) in A) else A[(last_tag, tag)] + 1
    # update n-gram counts for all orders of n-gram
    for i in range(len(ngram)):
        igram = ngram[i:]
        # Add count to emission probability matrix.
        B[(tag, igram)] = 1 if (not ((tag, igram) in B)) else B[(tag, igram)] + 1 
    return (Q, A, B, ngram, tag)


"""
Convert model of frequencies to probabilities.
"""
def freqs_to_probs(QAB):
    (Q,A,B) = QAB
    # Calculate emission probabilities from n-gram counts and state counts.
    for (tag, igram) in B:
        B[(tag, igram)] = B[(tag, igram)] / Q[tag]
    # Calculate transition probabilities from state transition counts and state counts.
    for (tag1, tag2) in A:
        A[(tag1, tag2)] = A[(tag1, tag2)] / Q[tag1]
    q_count = functools.reduce(lambda x,y: x+y, Q.values()) - Q[TAG_Q0] - Q[TAG_QF]
    if q_count != 0:
        for tag in Q:
            if tag != TAG_Q0 and tag != TAG_QF:
                Q[tag] = Q[tag]/q_count
    return (Q,A,B)

"""
Build model of frequencies from annotated file.
"""
def freqs_from_file(input_file, n):
    # Initialize state counts with start and end state counts set to zero.
    Q = { TAG_Q0 : 0, TAG_QF : 0 }
    # Initialize transition frequencies to empty dictionary.
    A = {}
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
                Q[TAG_Q0] += 1
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
    return (Q,A,B)

"""
main
"""

(Q,A,B) = freqs_to_probs(freqs_from_file(args.input_file, args.n))

print("***** Q *****")
for q in Q:
    print("Q[{0}] = {1}".format(q, Q[q]))
print("***** A *****\n")
for a in A:
    print("A[{0}] = {1}".format(a, A[a]))
print("***** B *****\n")
for b in B:
    print("B[{0}] = {1}".format(b, B[b]))
