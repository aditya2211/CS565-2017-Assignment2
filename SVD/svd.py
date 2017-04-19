import nltk
from collections import defaultdict
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
import pickle

import numpy as np
from numpy.linalg import norm

from random import normalvariate
from math import sqrt

f = open('text8')
data  =f.readlines()
tokens = data[0].split()
freqtable = nltk.FreqDist(tokens)
top10k = [word for (word,freq) in freqtable.most_common(10000)]
mapping = defaultdict(lambda: 'UNK')
for v in top10k:
    mapping[v] = v
tokens_with_UNK = [mapping[v] for v in tokens]

def build_vocab(tok):
    vocab = Counter()
    vocab.update(tok)
    return {word: (i, freq) for i, (word, freq) in enumerate(vocab.items())}


vocab_dict = build_vocab(tokens_with_UNK)

def build_cooccur(vocab, toks, window_size=5, min_count=None):
   

    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.items())

    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

    token_ids = [vocab[word][0] for word in toks]

    for center_i, center_id in enumerate(token_ids):
        context_ids = token_ids[max(0, center_i - window_size) : center_i]
        contexts_len = len(context_ids)

        for left_i, left_id in enumerate(context_ids):
            cooccurrences[center_id, left_id] += 1
            cooccurrences[left_id, center_id] += 1

    return cooccurrences

cooc_matrix = build_cooccur(vocab_dict,tokens_with_UNK)
#import pickle
#cooc_matrix = pickle.load(open('cooc_sparse.pickle','rb'))

cooc_matrix_dense = cooc_matrix.todense()
from sklearn.utils.extmath import randomized_svd


##SVD Using Library
U, Sigma, VT = randomized_svd(cooc_matrix, n_components=50,
                                      n_iter=10,
                                      random_state=99)


def randomUnitVector(n):
    unnormalized = [normalvariate(0, 1) for _ in range(n)]
    theNorm = sqrt(sum(x * x for x in unnormalized))
    return [x / theNorm for x in unnormalized]

##SVD Self Implementation
def mySVD1d(A, epsilon=1e-10):
   

    n, m = A.shape
    x = randomUnitVector(m)
    lastV = None
    currentV = x

    if n > m:
        B = np.dot(A.T, A)
    else:
        B = np.dot(A, A.T)

    iterations = 0
    while True:
        iterations += 1
        lastV = currentV
        currentV = np.dot(B, lastV)
        currentV = currentV / norm(currentV)

        if abs(np.dot(currentV, lastV)) > 1 - epsilon:
            print("converged in {} iterations!".format(iterations))
            return currentV


def mySVD(A, k=None, epsilon=1e-10):
   
    n, m = A.shape
    svdSoFar = []
    if k is None:
        k = min(n, m)

    for i in range(k):
        matrixFor1D = A.copy()

        for singularValue, u, v in svdSoFar[:i]:
            matrixFor1D -= singularValue * np.outer(u, v)

        v = mySVD1d(matrixFor1D, epsilon=epsilon)  # next singular vector
        u_unnormalized = np.dot(A, v)
        sigma = norm(u_unnormalized)  # next singular value
        u = u_unnormalized / sigma

        svdSoFar.append((sigma, u, v))

    singularValues, us, vs = [np.array(x) for x in zip(*svdSoFar)]
    return singularValues, us.T, vs

new_cooc = np.array(cooc_matrix_dense)
impSigma, impU, impV = mySVD(new_cooc,50)

pickle.dump(U,open("svd_library.pickle","wb"))
pickle.dump(impU,open("svd_selfimp.pickle","wb"))
pickle.dump(vocab_dict,open("svd_dict.pickle","wb"))
pickle.dump(cooc_matrix_dense,open("cooc_matrix.pickle","wb"))