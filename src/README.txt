hmmpos.py
=========
usage: hmmpos.py [-h] --training_file TRN --test_file TST [--n N] [--k K]
                 [--ngram_backoff | --ngram_weights]
                 [--heuristic_fallback | --no_heuristic_fallback]
                 [--heuristic_preseed | --no_heuristic_preseed]

Annotates a series of sentences with part-of-speech tags.

optional arguments:
  -h, --help            show this help message and exit
  --training_file TRN   3-column .conll training corpus
  --test_file TST       2-column .conll file (w/o POS tags) test corpus
  --n N                 highest order of n-gram to include in features
  --k K                 add-k smoothing constant
  --ngram_backoff       backs off emission probs on n-grams until 1-gram
  --ngram_weights       composites emission probs on n-grams with weights
                        [iteratively 0.75 of remainder until 1-gram]
  --heuristic_fallback  heuristic check during decoding for unknown tokens
  --no_heuristic_fallback
  --heuristic_preseed   heuristic preseed of emission probs for unknown tokens
  --no_heuristic_preseed

Competitions
============
Submitted under the handle 'ghostant' to COSC 6336 - Part-of-Speech Tagging with HMM CodaLab Competition.

Examples
========
Defaults

python3 ./hmmpos.py --training_file dataset/train.conll --test_file dataset/dev.conll [--n 1] [--k 0.01] [--ngram_weights] [--no_heuristic_fallback] [--no_heuristic_preseed]

Most accurate so far tested

python3 ./hmmpos.py --training_file dataset/train.conll --test_file dataset/dev.conll --n 4 --k 0.01 --ngram_weights --heuristic_preseed

Nearly as accurate but much faster

python3 ./hmmpos.py --training_file dataset/train.conll --test_file dataset/dev.conll --n 2 --k 0.01 --ngram_weights --heuristic_preseed

Saving Annotated Output

python3 ./hmmpos.py --training_file dataset/train.conll --test_file dataset/dev.conll --n 4 --k 0.01 --ngram_weights --heuristic_preseed > submission.txt

Best Practices
==============
Usage of the --heuristic_fallback option is discouraged since it is slower
than --heuristic_preseed and in all honesty no longer at parity in terms
of accuracy. It was a necessary step toward developing the --heuristic_preseed
option, and so it is of some interest in terms of the development history.

Using the defaults will yield fairly abysmal accuracy, so it is strongly
recommended to use the --heuristic_preseed option for best results.

Training File Format
====================
Training files must be in the .conll file format, i.e.,

<word>\t<language id>\t<POS tag>
<word>\t<language id>\t<POS tag>
...
<word>\t<language id>\t<POS tag>

<word>\t<language id>\t<POS tag>
<word>\t<language id>\t<POS tag>
...
<word>\t<language id>\t<POS tag>

Each observation is a row consisting of the word in question followed
by a tab, followed by an identification of the language to which the
word belongs, followed by the correct part-of-speech tag for the
observation.

Each sentence is composed of consecutive observations and divided from
the next by a blank line.

Constituents of acronyms should be separated by underscores (_)
so as to not be confused with punctuation.

Test File Format
================
The test file is in the same format as the training file except that
the part of speech tags and preceding tab spaces are not present.

Output Format
=============
The output of the program is the same as the training file format.

Known Bugs/Issues
=================
While some effort has been made to preserve probability mass in the employment
of heuristic pre-seeding, time constraints have prevented a thorough audit
heretofore. Some work needs to be done in this area.

add-k smoothing in the emission probability matrix has tended to favor
INTJ and some other classes for unknown tokens, i.e., tokens not seen
in training that are encountered in test, because of the initial equiprobable
distribution. Explicit stops were put in to avoid these errors, but
this leaves the tagger susceptible to not picking up new interjections,
etc., not encountered in training. A better solution is in order.

This tagger has only been tested on code-switched English/Spanish data.
There is no guarantee as to fitness for any other languages, particularly
languages with non-Latin orthography.

This program does NOT work with python2. Make sure python3 is the default
version or explicitly call the program as outlined in the examples.

Contact
=======
Please contact the author Michael Yantosca (mike@archivarius.net) via e-mail
with any questions, comments, or bug reports. 
