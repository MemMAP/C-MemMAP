#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Delegated Model Clustering

Created on Mon Dec 14 23:45:57 2020
@author: pengmiao
"""

from tensorflow.keras.models import load_model
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
X_all=[]
# Concatenate the weights of model layers as attributes of a trace
for app in TRACE_FILE_NAMES[0:]:
    print(app)
    path = "./Specialized_rerun_model/"
    model_ = load_model(path+app+"_t2t1.h5")
    X=[]
    #X=model_.layers[0].get_weights()[0].flatten()
    X=np.append(X,model_.layers[1].get_weights()[0].flatten())
    X=np.append(X,model_.layers[1].get_weights()[1].flatten())
    X=np.append(X,model_.layers[1].get_weights()[2].flatten())
    X=np.append(X,model_.layers[3].get_weights()[0].flatten())
    X=np.append(X,model_.layers[3].get_weights()[1].flatten())
    X_all.append(X)
X_all=np.array(X_all)    

#Dimension reduction using PCA
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(X_all)

kmeans = KMeans(n_clusters=3, random_state=6).fit(principalComponents)
print(kmeans.labels_)

data =principalComponents
label=kmeans.labels_
dict_cl={}
dict_cl[0]='purple'
dict_cl[1]='r'
dict_cl[2]='b'
dict_cl[3]='g'
dict_cl[4]='black'

ax = plt.subplot(111, projection='3d')  

for i in range(13):
    ax.scatter(data[i][0],data[i][1], data[i][2], c=dict_cl[label[i]],s=50.5,alpha=0.7) 

ax.set_zlabel('Z')  
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()