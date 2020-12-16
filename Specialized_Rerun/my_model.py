# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 21:33:30 2020

@author: pmzha
"""

import os
import numpy as np
#from Evaluate import Evaluate

from keras.models import Sequential, load_model
from keras.layers import Dense, Embedding, Dropout, LSTM
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import LSTM
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from keras.layers import CuDNNLSTM #use env:ef1 to train with GPU
from keras.callbacks import EarlyStopping

class my_model:
    def __init__(self, vocab_size, batch_size, embedding_dim, i_dim, o_dim):
        self.i_dim = i_dim
        self.o_dim = o_dim

        self.model = Sequential()
   #     self.model.add(Embedding(vocab_size, embedding_dim, input_length=i_dim))
        self.model.add(Embedding(vocab_size, embedding_dim, input_length=i_dim))
      #  self.model.add(LSTM(50))
       # self.model.add(LSTM(50,batch_input_shape=(1,1,10),stateful=True))
        self.model.add(CuDNNLSTM(50))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(20,activation='sigmoid'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(o_dim,activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model_name = 'DCLSTM'
        self.model.summary()

    def train(self,X_train, y_train, X_test, y_test, num_epochs, batch_size):
        history =self.model.fit(X_train,
                y_train,
                epochs=num_epochs,
                shuffle=False,
                batch_size=batch_size,
                validation_data=(X_test, y_test))
        return history
        
    def predict(self,X):
        return self.model.predict(np.asarray(X))
    
class my_model_stateful:
    def __init__(self, vocab_size, batch_size,embedding_dim, i_dim, o_dim):
        self.i_dim = i_dim
        self.o_dim = o_dim

        self.model = Sequential()
   #     self.model.add(Embedding(vocab_size, embedding_dim, input_length=i_dim))
        self.model.add(Embedding(vocab_size, embedding_dim, input_length=i_dim,
                       batch_input_shape=(batch_size,i_dim)))
       # self.model.add(LSTM(50))
       # (batch_size, timesteps, data_dim). 
        self.model.add(LSTM(50,batch_input_shape=(batch_size,i_dim,embedding_dim),stateful=True))
       # self.model.add(CuDNNLSTM(50))
        self.model.add(Dropout(0.1))
        self.model.add(Dense(20,activation='sigmoid'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(o_dim,activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        self.model_name = 'DCLSTM'
        self.model.summary()

    def train(self,X_train, y_train, X_test, y_test, num_epochs, batch_size):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        history =self.model.fit(X_train,
                y_train,
                epochs=num_epochs,
                shuffle=False,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping])
        return history
        
    def predict(self,X,batch_size):
        return self.model.predict(np.asarray(X),batch_size=batch_size)