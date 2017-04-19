# Faceless_Men CS565-Assignment2

## Neural Probabilistic Language Model

The model proposed by [Bengio et al.](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) is implemented in Tensorflow. It can be run using `nplm.py`.

Implementation details:
* Batch size = 5000
* Window size = 5
* Number of epochs = 5

The obtained embeddings are saved in *nplm_emb.pickle*

## Singular Value Decomposition

The file `svd.py` conatains the code for runnning the llibrary function for SVD and also the self implemented version of SVD. The code also generates the word-word co-occurence matrix. The embedding obtained using the library function are stored in a pickle file named *svd_library.pickle* and the embedding obtained through the self-implemented SVD are pickled in *svd_selfimp.pickle*.  

## Word2Vec

The CBOW model with negative sampling proposed by [Mikolov et al.](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) is implemented in Tensorflow. It can be run using `word2vec.py`.

For negative sampling, the **count** of all the words in the corpus is passed to the model, which then generates a unigram distribution and uses the inbuilt NCE loss function as follows.

```python
neg_sample = tf.nn.fixed_unigram_candidate_sampler(y_in, 1, neg_sample_size, True, dictionary_size, unigrams=count)
self.losses = tf.nn.nce_loss(W, b, tf.to_float(y_in), x_emb_in, neg_sample_size, dictionary_size, sampled_values=neg_sample)
```

Implementation details:
* Batch size = 5000
* Window size = 5
* Number of epochs = 50

The obtained embeddings are saved in *word2vec_emb.pickle*

### Named Entity Recognition (NER)

#### Generating the dataset

Required files:
* enwik8
* convert.pl
* extract.py

We use the [enwik8](http://mattmahoney.net/dc/textdata.html) corpus consisting of the first 10^8 words in the English language Wikipedia. This file contains complex XML encoding, so it is first converted to English sentences using a Perl script as `perl convert.pl enwik8 > text8`.

#### NER model

The data preprocessing, neural network implementation, model training and prediction generation is done in *entityrecognition.py*.

*entityrecognition.py* contains code for a single hidden layer model and a CNN model. In addition, an LSTM model was also implemented, but did not perform well.
