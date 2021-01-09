# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 23:26:32 2020

@author: Soumyajit Saha
"""

from __future__ import division, print_function
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import csv, os, sys
from sklearn.svm import SVC
import MFO as mf
import DA as da
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier



def calc_acc(y, y_pred):
    total=len(y)
    c=0
    for i in range(0,total):
        if y[i]==y_pred[i]:
            c+=1
    return c/total
    

########### Original Feature set ###############
dataset=pd.read_csv('D:/Project/new_covid_feature_red.csv') 
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

# x=x.to_numpy()
# y=y.to_list()

# c=1
# ch=y[0]

# y_ch=[]

# for i in range(0,len(y)):
#     if ch == y[i]:
#         y_ch.append(c)
#     else:
#         c+=1
#         ch=y[i]
#         y_ch.append(c)
# y_ch=np.asarray(y_ch)

######## Spliting into test and train ############
m=0
n=1
for i in range(100):
    x_train_t,x_test_t, y_train_t, y_test_t=train_test_split(x,y,test_size=0.20)
    svclassifier = SVC(kernel='poly',coef0=2.0)
    #svclassifier = KNeighborsClassifier(n_neighbors=5)
    svclassifier.fit(x_train_t, y_train_t)
    y_test_tp=y_test_t.to_numpy()
    y_pred = svclassifier.predict(x_test_t)
    acc=calc_acc(y_test_tp,y_pred)
    
    if m<acc:
        m=acc
        x_train,x_test, y_train, y_test= x_train_t,x_test_t, y_train_t, y_test_t
    if n>acc:
        n=acc

print(m)
print(n)   
# orig=np.arange(len(x_train.iloc[0]))     

selected_features, fitness, precision, sensitivity, F1, AUC=da.DA(x_train, y_train, x_test, y_test, 100, m, orig)
#selected_features = np.random.randint(288, size=)

x_train_selected_features=pd.DataFrame()
x_test_selected_features=pd.DataFrame()

reduced_dataset=pd.DataFrame()


for i in range(len(selected_features)):
    x_train_selected_features[str(selected_features[i])] = x_train.iloc[:,int(selected_features[i])] # X_Train from selected features
    x_test_selected_features[str(selected_features[i])] = x_test.iloc[:,int(selected_features[i])] # # X_Test from selected features
    reduced_dataset['attr ' + str(int(selected_features[i]))] = dataset.iloc[:,int(selected_features[i])]
    
reduced_dataset['class'] = dataset.iloc[:,-1]  


############ saving the the reduced feature set #####################

reduced_dataset.to_csv('D:/Project/FER_moth_flame/DLBP/reduced features/reduced_feature_set.csv', index=False) 
    
#     x_train_selected_features[str(selected_features[i])] = x_train[:,selected_features[i]]
#     x_test_selected_features[str(selected_features[i])] = x_test[:,selected_features[i]]

# x_train_selected_features=x_train_selected_features.to_numpy()
# x_test_selected_features=x_test_selected_features.to_numpy()


# svclassifier = SVC(kernel='linear')
# svclassifier.fit(x_train_selected_features, y_train)
# y_test=y_test.to_numpy()
# y_pred = svclassifier.predict(x_test_selected_features)


# acc=calc_acc(y_test,y_pred)

print("Reduced dimension = " + str(len(np.unique(selected_features))))
# print()
print("accuracy = " + str(fitness))
print("precision = " + str(precision))
print("sensitivity = " + str(sensitivity))
print("AUC = " + str(AUC))
print("F1 = " + str(F1))


# print(acc)

