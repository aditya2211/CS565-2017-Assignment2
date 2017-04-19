from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import pickle


##Select Method: 0 for NPLM, 1 for SVDLib, 2 for SVDSelf, 3 for CBOW
method=3

def plot_with_labels(TD_embeddings, labels, methodname):
  
  plt.figure(figsize=(20, 20))
  for i, label in enumerate(labels):
    x,y = TD_embeddings[i,:]
    plt.scatter(x, y)
    plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')

  filename=methodname+'tsne.svg'
  plt.savefig(filename)


def tsne(vector_embeddings,labels,method,num_words=100):
  
  tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
  TD_embeddings = tsne.fit_transform(vector_embeddings)
  plot_with_labels(TD_embeddings, labels, method)

def main():
  nplm_labels=pickle.load(open("nlpm_dict.pickle","rb"))
  svdl_labels=pickle.load(open("svd_dict.pickle","rb"))
  svds_labels=pickle.load(open("svd_dict.pickle","rb"))
  cbow_labels=pickle.load(open("word2vec_dict.pickle","rb"))
  all_labels=[nplm_labels,svdl_labels,svds_labels,cbow_labels]  
  labels=all_labels[method]
  revlabels = {i:word for word , (i,j) in labels.items()}
  
  nplm_embeddings=pickle.load(open("nlpm_emb.pickle","rb"))
  svdl_embeddings=pickle.load(open("svd_library.pickle","rb"))
  svds_embeddings=pickle.load(open("svd_selfimp.pickle","rb"))
  cbow_embeddings=pickle.load(open("word2vec_emb.pickle","rb"))
  all_embeddings=[nplm_embeddings,svdl_embeddings,svds_embeddings,cbow_embeddings]
  embeddings=all_embeddings[method]

  mnames=['nplm','svdlib','svdslf','word2vec']  

  #Random Words
  #random.seed(1)
  #favwords=random.sample(range(10000),100)

  #Frequent Words
  #topwords=pickle.load(open("T10K.pickle","rb"))[0:100]
  #favwords=[labels[i][0] for i in topwords]
  
  #Selected Words
  selwords=['zero','one','two','three','four','five','six','seven','eight','nine',
          'european','british','american','african','french','german','roman','japanese','chinese','italian',
          'football','cricket','hockey','tennis','baseball','basketball','soccer','golf','boxing','bowling',
          'swimming','running','eating','reading','playing','speaking','going','giving','writing','fighting',
          'january','february','march','april','june','july','september','october','november','december',
          'doctor','engineer','chemist','actor','farmer','lawyer','pilot','actress','director','musician',
          'school','college','university','academic','academy','institute','institutions','schools','colleges','academics',
          'hindu','muslim','christian','christianity','jewish','hinduism','islam','islamic','hindus','christians',
          'earthquake','flood','drought','storm','lightning','disaster','terrorism','bombing','blast','hazards',
          'computer','mobile','google','internet','website','bluetooth','software','microsoft','hardware','tablets']
  favwords=[labels[i][0] for i in selwords]
  
  
  tsne(embeddings[favwords,:],[revlabels[i] for i in favwords],mnames[method])

main()


    