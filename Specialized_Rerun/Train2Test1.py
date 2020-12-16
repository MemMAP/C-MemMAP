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
from dclstm_model import dclstm_model

PROJECT_ROOT_DIRECTORY = "./" 
tok_save_path=PROJECT_ROOT_DIRECTORY+"pk_file/"
np_save_path=PROJECT_ROOT_DIRECTORY+"np_file/"
model_save_path=PROJECT_ROOT_DIRECTORY+"model/"
look_back=3

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def Train2Test1(app_name, data_len_each, total_data_len, epoch):#app_name=blackscholes,-1,20
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
    '''
    source_path_list = sorted(glob.glob("/home/pengmiao/Project/PAKDD-2/my_meta/data_dt/"+app_name+"*.csv"))
    dataset_conc=f.Concatenate_files(source_path_list,data_len_each)
    print("concate done")  
    save_file_name="blackscholes"
    X_train, y_train, X_test, y_test = f.Tokenize_and_Binarize(dataset_conc,save_file_name,
                                                               tok_save_path,np_save_path,total_data_len,look_back)#total_data_len, lookback
    print()
    '''
    '''load np and token'''

    Data = np.load(np_save_path+app_name+"_np.npz")

    X_train, y_train, X_test, y_test=Data["X_train"],Data["y_train"],Data["X_test"],Data["y_test"]
    '''train model'''
    model_file_name=model_save_path+app_name+'_t2t1.h5'
    final_vocab_size=2
    embedding_dim=10
    i_dim=look_back*16
    o_dim=16
    batch_size=256
    num_epochs=epoch
    model_ = dclstm_model(final_vocab_size, batch_size,embedding_dim, i_dim, o_dim)
    history=model_.train(X_train, y_train,X_test, y_test, num_epochs, batch_size)
    model_.model.save(model_file_name)
    
    y_pred = model_.predict(X_test)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    aaaaa = f.convert_binary_to_dec(y_test)
    bbbbb = f.convert_binary_to_dec(y_pred)

    accuracy = accuracy_score(np.array(aaaaa), np.array(bbbbb))
    print("accuracy:",accuracy)
    return accuracy

TRACE_FILE_NAMES = [
        'blackscholes',
        'bodytrack',
        'canneal',
        'dedup',
        'facesim',
        'ferret',
        'fluidanimate',
        'freqmine',
        'raytrace',
        'streamcluster',
        'swaptions',
        'vips',
        'x264'
    ]
acc_ls= list()
for file in TRACE_FILE_NAMES[1:2]:
    print("App:",file)
    acc=Train2Test1(file,int(sys.argv[1]),-1,int(sys.argv[2]))#sequence lenth, epoch
    acc_ls.append(acc)
    print("App:",file,"; Acc:",acc)
    print("ACC_LIST:",acc_ls)
print(acc_ls)