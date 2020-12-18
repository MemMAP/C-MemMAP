
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

PROJECT_ROOT_DIRECTORY = "./" 
tok_save_path=PROJECT_ROOT_DIRECTORY+"pk_file/"
np_save_path=PROJECT_ROOT_DIRECTORY+"np_file/"
model_save_path=PROJECT_ROOT_DIRECTORY+"model/"
model_file_name=model_save_path+'T_all_t_each.h5'
look_back=3


source_path_list1 = sorted(glob.glob("../data_dt/*1_dt.csv"))
source_path_list2=sorted(glob.glob("../data_dt/*2_dt.csv"))
test_path_list=sorted(glob.glob("../data_dt/*3_dt.csv"))
source_path_list=source_path_list1+source_path_list2+test_path_list
print(source_path_list)


'''1. Concatenate files'''
data_len_each_file=int(sys.argv[1]) #200000
dataset_conc=f.Concatenate_files(source_path_list,data_len_each_file)

#source_path_list = sorted(glob.glob("/home/pengmiao/Project/PAKDD-2/my_meta/data_dt/blackscholes*.csv"))
#dataset_conc=f.Concatenate_files(source_path_list)
print("concate done")  

X_train, y_train, X_test, y_test=f.Tokenize_and_Binarize(dataset_conc,"T_all_t_each",tok_save_path,np_save_path,-1,look_back,0.3334)#data_len, lookback
