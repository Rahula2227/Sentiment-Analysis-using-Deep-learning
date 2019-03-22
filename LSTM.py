import numpy as np
import pandas as pd
import keras
import gensim
from keras.models import Sequential
from keras.layers import Dense,LSTM,GRU,Dropout
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
data=pd.read_csv('training.1600000.processed.noemoticon.csv', encoding="ISO-8859-1")
data=data[:1000000]
review=list()
lines= data['reviews'].values.tolist()
for line in lines:
  tokens=word_tokenize(line)
  tokens= [w.lower() for w in tokens]
  table=str.maketrans('','',string.punctuation)
  stripped =[w.translate(table) for w in tokens]
  words=[word for word in stripped if word.isalpha()]
  stop_words=set (stopwords.words('english'))
  words=[w for w in words if not w in stop_words]
  review.append(words)
EMBEDDING_DIM=100
model = gensim.models.Word2Vec (sentences=review,size=EMBEDDING_DIM,window=5,workers=4,min_count=1)
words=list(model.wv.vocab)
file='sentiments_word2vec.txt'
model.wv.save_word2vec_format(file,binary=False)
import os
embeddings_index={}
f=open(os.path.join('','sentiments_word2vec.txt'),encoding="utf-8")
for line in f:
  values=line.split()
  word=values[0]
  coefs=np.asarray(values[1:])
  embeddings_index[word]=coefs
f.close()  
max_length=max([len(s.split()) for s in data['reviews']])
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(review)
sequences=tokenizer_obj.texts_to_sequences(review)
word_index=tokenizer_obj.word_index
review_pad=pad_sequences(sequences,maxlen=max_length)
sentiments=data['target'].values
num_words=len(word_index)+1
embedding_matrix=np.zeros((num_words,EMBEDDING_DIM))
for word,i in word_index.items():
  if i>num_words:
    continue
  embedding_vector=embeddings_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i]=embedding_vector
    model1=Sequential()
    
embedding_layer=Embedding(num_words,EMBEDDING_DIM,embeddings_initializer=Constant(embedding_matrix),input_length=max_length,trainable=False)
model1.add(embedding_layer)
model1.add(LSTM(100))
model1.add(Dropout(0.2))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
validation_split=0.2
indices=np.arange(review_pad.shape[0])
np.random.shuffle(indices)
review_pad=review_pad[indices]
sentiments=sentiments[indices]
num_validation_samples=int(validation_split*review_pad.shape[0])
x_train_pad=review_pad[:-num_validation_samples]
y_train=sentiments[:-num_validation_samples]
x_test_pad=review_pad[-num_validation_samples:]
y_test=sentiments[-num_validation_samples:]
model1.fit(x_train_pad,y_train,batch_size=128,epochs=50,validation_data=(x_test_pad,y_test),verbose=2)
