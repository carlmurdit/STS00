# coding=utf8
#
# Copyright (c) 2012, Frane Saric
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
#   * If this software or its derivative is used to produce an academic
# publication, you are required to cite this work by using the citation
# provided on "http://takelab.fer.hr/sts".
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import math
from nltk.corpus import wordnet
import nltk
from collections import Counter, defaultdict
import sys
import re
import numpy
from numpy.linalg import norm

class Sim:
    def __init__(self, words, vectors):
        """
        :param words: 'nyt_words.txt'
        :param vectors: 'nyt_word_vectors.txt'
        """
        self.word_to_idx = {a: b for b, a in
                            enumerate(w.strip() for w in open(words))}
        self.mat = numpy.loadtxt(vectors)

    def bow_vec(self, b):
        """
        Generate a bag-of-words matrix
        :param b:
        :return:
        """
        vec = numpy.zeros(self.mat.shape[1])
        for k, v in b.iteritems():
            idx = self.word_to_idx.get(k, -1)
            if idx >= 0:
                vec += self.mat[idx] / (norm(self.mat[idx]) + 1e-8) * v
        return vec

    def calc(self, b1, b2):
        v1 = self.bow_vec(b1)
        v2 = self.bow_vec(b2)
        return abs(v1.dot(v2) / (norm(v1) + 1e-8) / (norm(v2) + 1e-8)) # normalise vectors

stopwords = set([
"i", "a", "about", "an", "are", "as", "at", "be", "by", "for", "from",
"how", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
"was", "what", "when", "where", "who", "will", "with", "the", "'s", "did",
"have", "has", "had", "were", "'ll"
])

nyt_sim = Sim('nyt_words.txt', 'nyt_word_vectors.txt')
wiki_sim = Sim('wikipedia_words.txt', 'wikipedia_word_vectors.txt')

def fix_compounds(a, b):
    # create a set of unique words in the 2nd sentence
    sb = set(x.lower() for x in b)

    # if compound in 1st sentence replace it (e.g. pay day becomes payday)
    a_fix = []
    la = len(a)
    i = 0
    while i < la:
        if i + 1 < la:
            comb = a[i] + a[i + 1]
            if comb.lower() in sb:
                a_fix.append(a[i] + a[i + 1])
                i += 2
                continue
        a_fix.append(a[i])
        i += 1
    return a_fix


def load_data(path):
    """
    :param path: str The filename
    :return: list(list(tuple(str, str)), list(tuple(str, str)))

        i.e. for each sentence_pair
            sentence_a(word, POS_tag), sentence_b(word, POS_tag)
    """

    sentences_pos = [] # initialise an empty list
    # remove <>, replace $US with $ in currencies
    r1 = re.compile(r'\<([^ ]+)\>')
    r2 = re.compile(r'\$US(\d)')
    for l in open(path):
        l = l.decode('utf-8')
        l = l.replace(u'’', "'")
        l = l.replace(u'``', '"')
        l = l.replace(u"''", '"')
        l = l.replace(u"—", '--')
        l = l.replace(u"–", '--')
        l = l.replace(u"´", "'")
        l = l.replace(u"-", " ")
        l = l.replace(u"/", " ")
        l = r1.sub(r'\1', l)
        l = r2.sub(r'$\1', l)
        s = l.strip().split('\t') # unused?

        # tokenise
        sa, sb = tuple(nltk.word_tokenize(s)
                          for s in l.strip().split('\t'))
        sa, sb = ([x.encode('utf-8') for x in sa],
                  [x.encode('utf-8') for x in sb])

        # replace contractions
        for s in (sa, sb):
            for i in xrange(len(s)):
                if s[i] == "n't":
                    s[i] = "not"
                elif s[i] == "'m":
                    s[i] = "am"

        # if a compound in one, use it, e.g. pay day becomes payday
        sa, sb = fix_compounds(sa, sb), fix_compounds(sb, sa)

        # POS Tagging
        #   pos_tag() creates a list(tuple(str, str)) for each sentence
        #       e.g. [('They', 'PRP'), ('refuse', 'VBP'), ('us', 'PRP')]
        sentences_pos.append((nltk.pos_tag(sa), nltk.pos_tag(sb)))

    return sentences_pos

def load_wweight_table(path):
    """
    :param path: str the filename ('word-frequencies.txt')
    :param n: int n-gram size
    :return: defaultdict[str,float]

    Uses Google Books Ngrams to get words and frequency (e.g. they 65081886).
    See http://takelab.fer.hr/sts/
    """

    lines = open(path).readlines()
    wweight = defaultdict(float) # create Dict with the word as the key
    if not len(lines):
        return (wweight, 0.) # default to 0 if file is empty
    totfreq = int(lines[0])
    for l in lines[1:]:
        w, freq = l.split()
        freq = float(freq)
        if freq < 10:
            continue # skip rare words
        wweight[w] = math.log(totfreq / freq)

    return wweight

# At program initialisation, look up the information content of words in the test set
# calculated from  Google Books Ngrams
wweight = load_wweight_table('word-frequencies.txt')
minwweight = min(wweight.values())

def len_compress(l):
    return math.log(1. + l)

to_wordnet_tag = {
        'NN':wordnet.NOUN,
        'JJ':wordnet.ADJ,
        'VB':wordnet.VERB,
        'RB':wordnet.ADV
    }

word_matcher = re.compile('[^0-9,.(=)\[\]/_`]+$')
def is_word(w):
    # true if only contains alpha characters
    return word_matcher.match(w) is not None

def get_locase_words(spos):
    """
    :param spos: list[tuple[str,str]] - A list of tuple[word, POS]
    :return: list[str]

    convert to lower case, excluding non-words and discarding POS info
    """

    return [x[0].lower() for x in spos
            if is_word(x[0])]

def make_ngrams(l, n):
    """
    :param l: List[str] A sentence
    :param n: int n-gram size
    :return: List[Tuple[str1, .... str_n]]

        e.g. if n==3, [('a', 'b', 'c'), ('b', 'c', 'd'), ('c', 'd', 'e')]
    """

    # e.g. if n == 3
    #   xrange(2) == 0,1
    #   a[start:end] is items start through end-1
    #   if end is -ve, it counts from the end of the array
    #       l[0:-3+0+1] == l[0:-2] == from 0 to last-2
    #       l[1:-3+1+1] == l[1:-1] == from 0 to last-1
    rez = [l[i:(-n + i + 1)] for i in xrange(n - 1)]

    # a[start:] means from start through to end of array
    # e.g. if n == 3
    #   rez.append(l[3-1:]) == rez.append(l[2:]) == starting c
    rez.append(l[n - 1:])

    # zip() converts n lists into a list of n-tuples
    # each list was created starting at a later point
    return zip(*rez)

def dist_sim(sim, la, lb):
    """
    :param sim: an instance of class Sim (words and vectors)
    :param la: a List[lemmas]
    :param lb: a List[lemmas]
    :return:
    """
    wa = Counter(la)
    wb = Counter(lb)
    d1 = {x:1 for x in wa}
    d2 = {x:1 for x in wb}
    return sim.calc(d1, d2)

def weighted_dist_sim(sim, lca, lcb):
    """
    :param sim: an instance of class Sim (words and vectors)
    :param lca: a List[lemmas]
    :param lcb: a List[lemmas]
    :return:
    """
    wa = Counter(lca)
    wb = Counter(lcb)
    wa = {x: wweight[x] * wa[x] for x in wa}
    wb = {x: wweight[x] * wb[x] for x in wb}
    return sim.calc(wa, wb)

def weighted_word_match(lca, lcb):
    """
    :param lca: List[str], non-lemmatised l-case words in a, stopwords included
    :param lcb: List[str], non-lemmatised l-case words in b, stopwords included
    :return: int or float, a similarity score

    See section 'Weighted Word Overlap' - uses WordNet and information content (IC)
    IC uses word-frequencies.txt generated from Google Books Ngrams
    """

    wa = Counter(lca)
    wb = Counter(lcb)
    wsuma = sum(wweight[w] * wa[w] for w in wa)
    wsumb = sum(wweight[w] * wb[w] for w in wb)
    wsum = 0.

    for w in wa:
        wd = min(wa[w], wb[w])
        wsum += wweight[w] * wd

    p = 0.
    r = 0.
    if wsuma > 0 and wsum > 0:
        p = wsum / wsuma
    if wsumb > 0 and wsum > 0:
        r = wsum / wsumb

    # Weighted Word Coverage, wwc(sent1, sent2) is the ratio of co-occurring words to words in sent2, weighted by IC
    # Weighted Word Overlap is the harmonic mean of p, the wwc(sent1, sent2) and r, the wwc(sent2, sent1)
    # The harmonic mean of precision and recall is the F1 measure (Kelleher p.476)
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return f1

wpathsimcache = {} # dictionary of scores already calculated Dict{Tuple(word_a, word_b), score}
def wpathsim(a, b):
    """
    :param a: a word
    :param b: a word
    :return: float, a similarity score

    Generates a similarity score using WordNet's basic basic path_similarity() function, which doesn't use depth
    """

    if a > b:
        # uses alphabetical order, to match any cached pairs
        b, a = a, b
    p = (a, b)
    if p in wpathsimcache:
        return wpathsimcache[p] # return cached score, if present
    if a == b:
        wpathsimcache[p] = 1.
        return 1.
    # Get all synsets for word a and word b (polysemous words will have more than one synset)
    sa = wordnet.synsets(a)
    sb = wordnet.synsets(b)
    # Choose the synsets that have the highest similarity
    # Uses NLTK Synset class' basic path_similarity() function, which doesn't use depth
    mx = max([wa.path_similarity(wb)
              for wa in sa
              for wb in sb
              ] + [0.])
    wpathsimcache[p] = mx # add to cache
    return mx

def calc_wn_prec(lema, lemb):
    """
    :param lema: list[str], lemmas of sentence a
    :param lema: list[str], lemmas of sentence b
    :return: int or float, a similarity score

    Generates a similarity score between lemmatised sentences using WordNet
    """

    rez = 0.
    for a in lema:
        ms = 0.
        for b in lemb:
            ms = max(ms, wpathsim(a, b)) # find the most similar lemma in lemb for the current lemma in lema
        rez += ms # increment the score before moving on to the next lemma in lema
    return rez / len(lema) # decrease the score in proportion to the length of lema

def wn_sim_match(lema, lemb):
    """
    :param lema: List[str], lemmas of sentence a
    :param lema: List[str], lemmas of sentence b
    :return: int or float, a similarity score

    'WordNet-Augmented Word Overlap' - paper
    Generates a similarity score between lemmatised sentences using WordNet.
    Doesn't use Leacock & Chodorow or Lin measures as reported in their paper.
    Compares lema to lemb, then lemb to lema and combines them using an undocumented weighting system.
    """

    f1 = 1.
    p = 0.
    r = 0.
    if len(lema) > 0 and len(lemb) > 0:
        p = calc_wn_prec(lema, lemb)
        r = calc_wn_prec(lemb, lema)
        f1 = 2. * p * r / (p + r) if p + r > 0 else 0.
    return f1

def ngram_match(sa, sb, n):
    """
    :param sa: List[str], words in a
    :param sb: List[str], words in b
    :param n: int, n-gram size
    :return: int or float, a similarity score
    Creates n-grams, counts overlapping words
    """

    nga = make_ngrams(sa, n)
    ngb = make_ngrams(sb, n)
    matches = 0

    # A Counter keeps track of how many times each word is found in Sentence a
    c1 = Counter(nga)
    for ng in ngb:
        if c1[ng] > 0:
            # if word in an n-gram of b found in an n-gram of a
            # remove word from counter so it isn't matched again if Sentence b has the word twice
            c1[ng] -= 1
            # increment the overlap counter
            matches += 1
    p = 0.
    r = 0.
    f1 = 1.
    if len(nga) > 0 and len(ngb) > 0:
        p = matches / float(len(nga)) # ratio of words in a that match b
        r = matches / float(len(ngb)) # ratio of words in b that match a
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.
    return f1

def get_lemmatized_words(sa):
    """
    :param sa: List[Tuple(str,str)], a list of words with their POS
    :return: List[str], a list of lemmas (POS info is discarded)
    """
    rez = []
    for w, wpos in sa:

        w = w.lower()
        if w in stopwords or not is_word(w):
            continue

        # convert POS to WordNet tag (2nd of tuple), e.g. 'NN' becomes wordnet.NOUN
        wtag = to_wordnet_tag.get(wpos[:2])
        if wtag is None:
            # if no POS tag, the lemma is the unchanged word
            wlem = w
        else:
            # Look up forms not in WordNet, with the help of Morphy (www.nltk.org/howto/wordnet.html)
            # wn.morphy('denied', wn.VERB) returns deny
            wlem = wordnet.morphy(w, wtag) or w
        rez.append(wlem)

    # return a list of lemmas (POS info is discarded)
    return rez

def is_stock_tick(w):
    return w[0] == '.' and len(w) > 1 and w[1:].isupper()

def stocks_matches(sa, sb):
    """
    :param sa: List[str], a list of words in a
    :param sb: List[str], a list of words in b
    :return: Tuple[float, float]

        a log of counts of stock tokens found in both

        a similarity score (1 means every stock id is in both)
    """

    ca = set(x[0] for x in sa if is_stock_tick(x[0]))
    cb = set(x[0] for x in sb if is_stock_tick(x[0]))
    isect = len(ca.intersection(cb))
    la = len(ca)
    lb = len(cb)

    f = 1.
    if la > 0 and lb > 0:
        if isect > 0:
            p = float(isect) / la
            r = float(isect) / lb
            f = 2 * p * r / (p + r)
        else:
            f = 0.

    f += case_matches(sa, sb)
    return (len_compress(la + lb), f)

def case_matches(sa, sb):
    ca = set(x[0] for x in sa[1:] if x[0][0].isupper()
            and x[0][-1] != '.')
    cb = set(x[0] for x in sb[1:] if x[0][0].isupper()
            and x[0][-1] != '.')
    la = len(ca)
    lb = len(cb)
    isect = len(ca.intersection(cb))

    f = 1.
    if la > 0 and lb > 0:
        if isect > 0:
            p = float(isect) / la
            r = float(isect) / lb
            f = 2 * p * r / (p + r)
        else:
            f = 0.

    # return a 2-tuple
    #       len_compress() uses log of counts of capitalised tokens found in the pair
    #       f is a similarity score (1 for any capitised words are in both)
    return (len_compress(la + lb), f)

# regex for checking if it excludes numeric characters
risnum = re.compile(r'^[0-9,./-]+$')
# regex for checking if it contains a digit
rhasdigit = re.compile(r'[0-9]')

def match_number(xa, xb):
    if xa == xb:
        return True
    xa = xa.replace(',', '')
    xb = xb.replace(',', '')

    try:
        va = int(float(xa))
        vb = int(float(xb))
        if (va == 0 or vb == 0) and va != vb:
            return False
        fxa = float(xa)
        fxb = float(xb)
        if abs(fxa - fxb) > 1:
            return False
        diga = xa.find('.')
        digb = xb.find('.')
        diga = 0 if diga == -1 else len(xa) - diga - 1
        digb = 0 if digb == -1 else len(xb) - digb - 1
        if diga > 0 and digb > 0 and va != vb:
            return False
        dmin = min(diga, digb)
        if dmin == 0:
            if abs(round(fxa, 0) - round(fxb, 0)) < 1e-5:
                return True
            return va == vb
        return abs(round(fxa, dmin) - round(fxb, dmin)) < 1e-5
    except:
        pass

    return False

def number_features(sa, sb):
    """
    :param sa: List[str], a list of words in a
    :param sb: List[str], a list of words in b
    :return: Tuple[float, float, float]

      a log of counts of numbers in the pair

      a similarity score (1 for every number is in both)

      subset: 1 if all numbers matched, else 0
    """

    # create a set of the numeric tokens in each sentence
    numa = set(x[0] for x in sa if risnum.match(x[0]) and
            rhasdigit.match(x[0]))
    numb = set(x[0] for x in sb if risnum.match(x[0]) and
            rhasdigit.match(x[0]))

    isect = 0  # counter for numbers found in both sentences
    for na in numa:
        if na in numb:
            isect += 1
            continue
        for nb in numb:
            if match_number(na, nb):
                isect += 1
                break

    # count the numeric tokens for each sentence
    la, lb = len(numa), len(numb)

    f = 1.
    subset = 0.
    if la + lb > 0:
        if isect == la or isect == lb:
            # if all number in a or b are matched
            subset = 1.
        if isect > 0:
            # some match, some don't; use % of each sentence matched
            p = float(isect) / la
            r = float(isect) / lb
            f = 2. * p * r / (p + r)
        else:
            # no matches
            f = 0.

    # return a 3-tuple
    #   len_compress() uses log of counts of numbers in the pair
    #   f is a similarity score (1 for any numbers are in both)
    #   subset is 1 if all numbers matched, else 0
    return (len_compress(la + lb), f, subset)

def relative_len_difference(lca, lcb):
    """
    :param lca: lcase words in sent_a, without stopwords
    :param lcb: lcase words in sent_b, without stopwords
    :return: Float, a similarity score

    Get ratio of sentence difference to the maximum sentence length
    """
    la, lb = len(lca), len(lcb)
    return abs(la - lb) / float(max(la, lb) + 1e-5)

def relative_ic_difference(lca, lcb):
    #wa = sum(wweight[x] for x in lca)
    #wb = sum(wweight[x] for x in lcb)
    wa = sum(max(0., wweight[x] - minwweight) for x in lca)
    wb = sum(max(0., wweight[x] - minwweight) for x in lcb)
    return abs(wa - wb) / (max(wa, wb) + 1e-5) # sci notation: 1 * 10 ^ -5

def calc_features(sa, sb):
    """
    :param sa: List(Tuple(str, str)), Sentence a as List(word, POS_tag)
    :param sb: List(Tuple(str, str)), Sentence b as List(word, POS_tag)
    :return: List[] a list of features for the sentence pair instance

    Generates features using various similarity measures
    """
    # convert to lower case, excluding non-words and discarding POS info
    olca = get_locase_words(sa)
    olcb = get_locase_words(sb)

    # remove stopwords
    lca = [w for w in olca if w not in stopwords]
    lcb = [w for w in olcb if w not in stopwords]

    # get List(str) of lemmas discarding POS info
    lema = get_lemmatized_words(sa)
    lemb = get_lemmatized_words(sb)

    # create a list of features for each pair
    f = []

    # number_features() returns Tuple[float, float, float], i.e. 3 features
    f += number_features(sa, sb)

    # case_matches() returns Tuple[float, float], i.e. 2 features
    f += case_matches(sa, sb)

    # stocks_matches() returns Tuple[float, float], i.e. 2 features
    f += stocks_matches(sa, sb)

    f += [
            ngram_match(lca, lcb, 1),  # make features from n-grams of words
            ngram_match(lca, lcb, 2),
            ngram_match(lca, lcb, 3),
            ngram_match(lema, lemb, 1), # make features from  n-grams of lemmas
            ngram_match(lema, lemb, 2),
            ngram_match(lema, lemb, 3),
            wn_sim_match(lema, lemb), # 'WordNet-Augmented Word Overlap' finds synonyms in WordNet
            weighted_word_match(olca, olcb), # 'Weighted Word Overlap' using WordNet and information content (IC)
            weighted_word_match(lema, lemb),
            dist_sim(nyt_sim, lema, lemb), # 200 dim word-vector mapping from New York Times Annotated Corpus (NYT)
            #dist_sim(wiki_sim, lema, lemb),
            weighted_dist_sim(nyt_sim, lema, lemb), # word-vector mapping from NYT weighting with IC
            weighted_dist_sim(wiki_sim, lema, lemb), # 500 dim word-vector mapping from Wikipedia weighting with IC
            relative_len_difference(lca, lcb),
            relative_ic_difference(olca, olcb)
        ]

    return f # 21 features total

if __name__ == "__main__":
    # verify 1 (for test) or 2 (for train) parameters present, e.g.
    # python takelab_simple_features.py train/STS.input.MSRvid.txt train/STS.gs.MSRvid.txt > msrvid-train.txt
    # python takelab_simple_features.py test/STS.input.MSRvid.txt > msrvid-test.txt
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print >>sys.stderr, "Usage: "
        print >>sys.stderr, "  %s input.txt [scores.txt]" % sys.argv[0]
        exit(1)

    scores = None
    if len(sys.argv) >= 3:
        # read gold standard file if training a model
        scores = [float(x) for x in open(sys.argv[2])]

    # load_data() returns a list(list(tuple(str, str)), list(tuple(str, str)))
    #   i.e. for each sentence_pair
    #       (sentence_A(word, POS_tag), sentence_A(word, POS_tag))
    # enumerate() returns tuples (index, value) obtained from an iterable.
    for idx, sp in enumerate(load_data(sys.argv[1])):

        # if gold standard file supplied, merge it with the training data, else use 0
        y = 0. if scores is None else scores[idx]

        # print gold standard, 1-based index and x (the numbered features) for each pair
        # 5.0 1:0.000000 2:1.000000 3:0.000000 4:0.000000 5:1.000000 6:0.000000 ... 20:0.000000 21:0.005196
        print y, ' '.join('%d:%f' % (i + 1, x) for i, x in
                          enumerate(calc_features(*sp)))  # if x)
