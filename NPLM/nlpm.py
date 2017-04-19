from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import zipfile
import numpy as np
import pickle
import tensorflow as tf
import nltk
from collections import defaultdict
from collections import Counter
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

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

class part1NPLM(object):

    def __init__(self, dictionary_size, emb_size=50, window_size=5, hidden_layer_size=100, learning_rate=0.01, l2_reg_lambda=0.01):

        self.input  = tf.placeholder(tf.int32, [None,window_size], name="input")
        self.output = tf.placeholder(tf.int32, [None], name="output")
        
        # Initialization
        self.W_emb =  tf.Variable(tf.random_uniform([dictionary_size, emb_size], -1.0, +1.0))
        
        # Embedding layer
        x_emb = tf.nn.embedding_lookup(self.W_emb, self.input)
        x_emb = tf.reshape(x_emb,[-1,window_size*emb_size])
        
        y_one_hot = tf.one_hot(self.output,dictionary_size)

        # Fully connetected layer
        H = tf.Variable(tf.truncated_normal([window_size*emb_size, hidden_layer_size], stddev=0.1), name="H")
        d = tf.Variable(tf.constant(0.1, shape=[hidden_layer_size]), name="d")
        h1 = tf.tanh(tf.nn.xw_plus_b(x_emb, H, d))

        #Regression layer
        U = tf.Variable(tf.truncated_normal([hidden_layer_size, dictionary_size], stddev=0.1), name="U")
        b = tf.Variable(tf.constant(0.1, shape=[dictionary_size]), name="b")
        W = tf.Variable(tf.truncated_normal([window_size*emb_size,dictionary_size],stddev=0.1), name="W")

        h2 = tf.add(tf.nn.xw_plus_b(h1, U, b),tf.matmul(x_emb,W))


        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(H)
        l2_loss += tf.nn.l2_loss(U)
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(W_emb)


        # prediction and loss function
        self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=h2, labels=y_one_hot)
        self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda*l2_loss

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=session_conf)  

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def train_step(self, in_batch, out_batch):
        feed_dict = {
                self.input: in_batch,
                self.output: out_batch
                        }
        _,loss, W_e = self.sess.run([self.optimizer, self.loss, self.W_emb], feed_dict)
            # print ("step "+str(step) + " loss "+str(loss))
        return W_e,loss


dictionary = dict((word, i) for word, (i, _) in vocab_dict.items())
reverse_dictionary = dict((i, word) for word, (i, _) in vocab_dict.items())
data = [vocab_dict[word][0] for word in tokens_with_UNK]
#count = pickle.load(handle)


nplm = part1NPLM(len(dictionary))
num_epochs = 1
num_train = len(data) - 4
batch_size = 512
num_batches = num_train//batch_size
window = 5

W = []

for j in range(num_epochs):
    for i in range(num_batches):
        x_batch = []
        y_batch = []
        for k in range(batch_size):
            temp_list = []
            for p in range(window):
                temp_list.append(data[i*batch_size+k+p])
        
            x_batch.append(temp_list)	
            y_batch.append(data[i*batch_size+k+window])

        W, loss = nplm.train_step(x_batch,y_batch)
        print("Epoch",j+1,"Batch",i+1,"Loss",loss)
        
pickle.dump(W,open("nplm_emb.pickle","wb"))
pickle.dump(vocab_dict,open("nlpm_dict.pickle","wb"))

