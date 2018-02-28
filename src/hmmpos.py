
"""
model = set of n-gram models
ngram = current n-gram
line = line to parse and add to the model
n = greatest n-gram order
"""
def add_word_lang_tag(Q, A, B, n, ngram, last_tag, word, lang, tag):
    if len(ngram) == n:
        ngram = ngram[1:]
    ngram = ngram + ((word, lang),)
    # Increment population count of tag.
    Q[tag] = 1 if (not (tag in Q)) else Q[tag] + 1
    # Increment count of last_tag -> tag in transition probability matrix.
    A[(last_tag, tag)] = 1 if not ((last_tag, tag) in A) else A[(last_tag, tag)] + 1
    # update n-gram counts for all orders of n-gram
    for i in range(n):
        igram = ngram[i:]
        # Add count to emission probability matrix.
        B[(tag, igram)] = 1 if (not ((tag, igram) in B)) else B[(tag, igram)] + 1 
    return (Q, A, B, ngram, tag)

input_file = "dataset/dev.conll"
Q = {}
A = {}
B = {}
n = 2
ngram = ()
last_tag = ""

# https://docs.python.org/3/library/functions.html#open
with open(input_file) as fp:
    for line in iter(fp.readline, ''):
        line = line.strip()
        if len(line) == 0:
            ngram = ()
            last_tag = ""
        else:
            (word, lang, tag) = tuple(line.split('\t')[0:3])
            (Q, A, B, ngram, last_tag) = add_word_lang_tag(Q, A, B, n, ngram, last_tag, word, lang, tag)

print("***** Q *****")
for q in Q:
    print("Q[{0}] = {1}".format(q, Q[q]))
print("***** A *****\n")
for a in A:
    print("A[{0}] = {1}".format(a, A[a]))
print("***** B *****\n")
for b in B:
    print("B[{0}] = {1}".format(b, B[b]))
