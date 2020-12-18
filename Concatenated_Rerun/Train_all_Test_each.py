import functions as f
import pickle
from sklearn.model_selection import train_test_split
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical, Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.utils import plot_model
from keras.models import load_model, save_model
from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import keras.backend as K
from sys import getsizeof
import statsmodels.api as sm
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from collections import Counter
import numpy as np
from numpy import argmax
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.rcParams['text.usetex'] = False
# matplotlib.use('Agg')
import time
import sys
from utils import *
import inspect
import collections
import glob
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from dclstm_model import dclstm_model
import numpy as np

PROJECT_ROOT_DIRECTORY = "./" 
tok_save_path=PROJECT_ROOT_DIRECTORY+"pk_file/"
np_save_path=PROJECT_ROOT_DIRECTORY+"np_file/"
model_save_path=PROJECT_ROOT_DIRECTORY+"model/"
model_file_name=model_save_path+'T_all_t_each.h5'
look_back=3

data_len_each_file=int(sys.argv[1])

def Train_all_Test_1(np_save_path, total_data_len, epoch):#app_name=blackscholes,-1,20
    '''
    input: 
        app_name;
        data_len_each: 200k, in each file, use -1 for all
        total_data_len: after concat, select part of it , for develop, use -1
        epoch: train epoch
    output:
        model test accuracy
    saved:
        np, pk, model
    '''
    '''1. Concatenate files'''
    
    Data = np.load(np_save_path+"T_all_t_each_np.npz")
   #  np.load = np_load_old#label
    
    X_train, y_train, X_test, y_test=Data["X_train"],Data["y_train"],Data["X_test"],Data["y_test"]
    '''train model'''
    final_vocab_size=2
    embedding_dim=10
    i_dim=look_back*16
    o_dim=16
    batch_size=256
    num_epochs=epoch
    model_ = dclstm_model(final_vocab_size, batch_size,embedding_dim, i_dim, o_dim)
    history=model_.train(X_train, y_train,X_test, y_test, num_epochs, batch_size)
    model_.model.save(model_file_name)


    fig1 = plt.figure(dpi=50, figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel("Epoch")
    #plt.ylabel("Training loss")
    plt.legend(loc="best")
    fig2 = plt.figure(dpi=50, figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuray')
    plt.xlabel("Epoch")
    plt.legend(loc="best")

    TRACE_FILE_NAMES = [
            'blackscholes',
            'bodytrack',
            'canneal_',
            'dedup',
            'facesim',
            'ferret',
            'fluidanimate',
            'freqmine',
            'raytrace',
            'streamcluster',
            'swaptions',
            'vips_',
            'x264'
        ]
    for i in range(13):
        print("acc:",TRACE_FILE_NAMES[i])
        X_test_piece=X_test[data_len_each_file*i:data_len_each_file*(i+1)+1]
        y_test_piece=y_test[data_len_each_file*i:data_len_each_file*(i+1)+1]
       # print(X_test_piece,y_test_piece)
        y_pred = model_.predict(X_test_piece)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        aaaaa = f.convert_binary_to_dec(y_test_piece)
        bbbbb = f.convert_binary_to_dec(y_pred)
        accuracy = accuracy_score(np.array(aaaaa), np.array(bbbbb))
        print("accuracy:",accuracy)
        
if __name__ == "__main__":
    # execute only if run as a script
    Train_all_Test_1(np_save_path,-1,int(sys.argv[2]))#epoch, 20
   # print("App:",file,"; Acc:",acc)
