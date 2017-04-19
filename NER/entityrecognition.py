from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from collections import Counter

## Select Method : 0 for NLPM, 1 for SVDLib, 2 for CBOW
method=0
## Select Whether to Update or Not
update=False

#Data Preparation
def InputDataPrep(data):

    for i in range(data.shape[0]):
        if (str(data.WORD[i]).lower() in labels.keys()):
            data.WORD[i]=data.WORD[i].lower()
        else:
            data.WORD[i]='UNK'

    window = 2
    X = []
    for i in range(data.shape[0]):
        temp_list=[]
        for p in range(-1*window,window+1):
            t=i+p
            t=max(0,t)
            t=min(t,data.shape[0]-1)
            temp_list.append(labels[data.WORD[t]][0])
        
        X.append(temp_list)

    return X

def argmaxfunc(predictions,ratio,iter=10,alpha=1):
    weight=np.ones(len(numtag))
    mdfct=100
    finaloutput=np.argmax(predictions,1)
    for i in range(iter):
        modpred=[]
        for item in predictions:
            tmp=np.multiply(weight,item)
            tmp=tmp/np.sum(tmp)
            modpred+=[list(tmp)]
        modpred=np.array(modpred)
    
        output=np.argmax(modpred,1)
        frat=[]
        for i in range(len(numtag)):
            frat+=[Counter(output)[i]]
        frat=np.array(frat)
        frat=frat/np.sum(frat)
        
        dfct=frat-ratio
        totd=np.sum(np.abs(dfct))
        print(totd,weight)
        
        if(totd<mdfct):
            mdfct=np.sum(np.abs(dfct))
            finaloutput=output
        
        for i in range(len(weight)):
            weight[i]=max(weight[i]-min(alpha*dfct[i],1),0)
        predictions=modpred
    
    return finaloutput        
    
def evaluator(obs,pred):
    N1=np.count_nonzero(obs)
    N2=np.count_nonzero(pred)
    N3=np.sum((obs==pred)&(obs!=0))
    
    prec=N3/N2
    rec=N3/N1
    f1=(2*prec*rec)/(prec+rec)
    acc=np.mean(obs==pred)
    
    return acc,prec,rec,f1
    
def nanreplace(inp):
    for i in range(len(inp)):
        if str(inp[i])=='nan':
            inp[i]='O'
    return inp
            
    
class NER(object):

    def __init__(self, dictionary_size, vect ,emb_size=50, window_size=5, hidden_layer_size=100, learning_rate=0.01, l2_reg_lambda=0.01, output_size=8):

        self.input  = tf.placeholder(tf.int32, [None,(window_size)], name="input")
        self.output = tf.placeholder(tf.int32, [None], name="output")
        
        #Initialization
        self.W_emb =  tf.Variable(vect,trainable=update)
        
        #Embedding layer
        x_emb = tf.nn.embedding_lookup(self.W_emb, self.input)
        x_emb = tf.reshape(x_emb,[-1,window_size*emb_size])
        
        y_one_hot = tf.one_hot(self.output,output_size)

        #Fully connetected layer
        H = tf.Variable(tf.truncated_normal([window_size*emb_size, hidden_layer_size], stddev=0.1), name="H")
        d = tf.Variable(tf.constant(0.1, shape=[hidden_layer_size]), name="d")
        h1 = tf.tanh(tf.nn.xw_plus_b(x_emb, H, d))

        #Regression layer
        U = tf.Variable(tf.truncated_normal([hidden_layer_size, output_size], stddev=0.1), name="U")
        b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="b")

        h2 = tf.nn.xw_plus_b(h1, U, b)
        self.h2sm = tf.nn.softmax(h2)

        #Regularization
        l2_loss = tf.constant(0.0)
        l2_loss += tf.nn.l2_loss(H)
        l2_loss += tf.nn.l2_loss(U)

        #Prediction and Loss Function
        self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=h2, labels=y_one_hot)
        self.loss = tf.reduce_mean(self.losses) + l2_reg_lambda*l2_loss

        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        self.sess = tf.Session(config=session_conf)  

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def train_step(self, in_batch, out_batch):
        feed_dict = {self.input: in_batch, self.output: out_batch}
        _,loss, W_embn = self.sess.run([self.optimizer, self.loss, self.W_emb], feed_dict)
            # print ("step "+str(step) + " loss "+str(loss))
        return W_embn,loss
    
    def prediction(self, in_batch, out_batch):
        feed_dict = {self.input: in_batch, self.output: out_batch}
        pred=self.sess.run(self.h2sm, feed_dict)
        return pred


nlpm_labels=pickle.load(open("nlpm_dict.pickle","rb"))
svd_labels=pickle.load(open("svd_dict.pickle","rb"))
cbow_labels=pickle.load(open("word2vec_dict.pickle","rb"))
all_labels=[nlpm_labels,svd_labels,cbow_labels]

nlpm_embeddings=pickle.load(open("nlpm_emb.pickle","rb"))
svd_embeddings=pickle.load(open("svd_library.pickle","rb"))
cbow_embeddings=pickle.load(open("word2vec_emb.pickle","rb"))
all_embeddings=[nlpm_embeddings,svd_embeddings,cbow_embeddings]

labels=all_labels[method]
vector_embeddings=np.float32(all_embeddings[method])

train_data=pd.read_csv('train.tsv',sep=" ",header=None)
train_data.columns=['WORD','V1','V2','TAG']
val_data=pd.read_csv('val.tsv',sep=" ",header=None)
val_data.columns=['WORD','V1','V2','TAG']
test_data=pd.read_csv('test.tsv',sep=" ",header=None)
test_data.columns=['WORD','V1','V2','TAG']

train_X=InputDataPrep(train_data)
val_X=InputDataPrep(val_data)
test_X=InputDataPrep(test_data)
train_data.TAG=nanreplace(train_data.TAG)
val_data.TAG=nanreplace(val_data.TAG)
test_data.TAG=nanreplace(test_data.TAG)
train_y=list(train_data.TAG)
val_y=list(val_data.TAG)
test_y=list(test_data.TAG)

tagnum={'O':0,'I-PER':1,'I-LOC':2,'I-ORG':3,'I-MISC':4,'B-LOC':5,'B-ORG':6,'B-MISC':7}
numtag={0:'O',1:'I-PER',2:'I-LOC',3:'I-ORG',4:'I-MISC',5:'B-LOC',6:'B-ORG',7:'B-MISC'}

train_y= [tagnum[str(i)] for i in train_y]
val_y=[tagnum[str(i)] for i in val_y]
test_y=[tagnum[str(i)] for i in test_y]
               
ratio=np.array(list(Counter(train_y).values()))
ratio=ratio/np.sum(ratio)

#Parameters               
nplm = NER(len(labels),vector_embeddings)
num_epochs = 5
num_train = train_data.shape[0]
batch_size = 256
num_batches = num_train//batch_size
window = 2

#Training and Validation
for j in range(num_epochs):
    for i in range(num_batches):
        x_batch = train_X[(i*batch_size):((i+1)*batch_size)]
        y_batch = train_y[(i*batch_size):((i+1)*batch_size)]

        W_embn, loss = nplm.train_step(x_batch,y_batch)
        print("Epoch",j+1,"Batch",i+1,"Loss",loss)

    train_predictions = nplm.prediction(train_X,train_y)
    #train_output = np.argmax(train_predictions,1)
    train_output = argmaxfunc(train_predictions,ratio)
    train_accuracy,train_prec,train_recall,train_f1 = evaluator(np.array(train_y),np.array(train_output))
    
    val_predictions = nplm.prediction(val_X,val_y)
    #val_output = np.argmax(val_predictions,1)
    val_output = argmaxfunc(val_predictions,ratio)
    val_accuracy,val_prec,val_recall,val_f1 = evaluator(np.array(val_y),np.array(val_output))
    
print("Epoch",j+1,"Loss",loss,"Train_F1",train_f1,"Val_F1",val_f1)
    

#Testing
test_predictions = nplm.prediction(test_X,test_y)
#test_output = np.argmax(test_predictions,1)
test_output=argmaxfunc(test_predictions,ratio)
test_accuracy,test_prec,test_recall,test_f1 = evaluator(np.array(test_y),np.array(test_output))

#Output File Generation for Test Set - File named based on embeddings used
mnames=['nplm','svd','word2vec']
filename='output'+mnames[method]+str(update)+'.txt'
test_data=pd.read_csv('test.tsv',sep=" ",header=None)
test_data.columns=['WORD','V1','V2','TAG']
test_data.TAG=nanreplace(test_data.TAG)
test_output_labels=[numtag[i] for i in test_output]
test_output_labels=pd.DataFrame(test_output_labels)
test_final=pd.concat([test_data[['WORD','V1','TAG']],test_output_labels],axis=1)
test_final.to_csv(filename,header=None,sep=" ",index=None)
