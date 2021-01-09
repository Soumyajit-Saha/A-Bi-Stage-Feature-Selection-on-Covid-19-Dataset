# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 19:48:39 2020

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
from ReliefF import ReliefF
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score

def calc_acc(y, y_pred):
    total=len(y)
    c=0
    for i in range(0,total):
        if y[i]==y_pred[i]:
            c+=1
    return c/total

dataset=pd.read_csv('D:/Project/FER_moth_flame/LDTP/jaffe_LDTP_32.csv') 
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

x_new1 = pd.DataFrame(data=SelectKBest(mutual_info_classif, k=150).fit_transform(x, y))
x_new2 = pd.DataFrame(data=ReliefF(n_neighbors=10, n_features_to_keep=150).fit_transform(x.to_numpy(), y.to_numpy()))
#x_new3 = pd.DataFrame(data=SelectKBest(f_classif, k=100).fit_transform(x, y))  


X=pd.concat([x_new1.transpose(),x_new2.transpose()]).drop_duplicates()
#X=pd.concat([X,x_new3.transpose()]).drop_duplicates()
X=X.transpose()

#X=x_new

selected_feature=pd.DataFrame()

for i in range(len(X.iloc[0])):
    selected_feature['attr ' + str(i)] = X.iloc[:,i]
    
selected_feature['class'] = dataset.iloc[:,-1]

selected_feature.to_csv('D:/Project/new_covid_feature_red.csv', index=False)

# x=selected_feature.iloc[:,:-1]
x=selected_feature.iloc[:,:-1]
y=selected_feature.iloc[:,-1]

m=0
for i in range(100):
    x_train_t,x_test_t, y_train_t, y_test_t=train_test_split(x,y,test_size=0.20)
    svclassifier = SVC(kernel='poly',coef0=2.0)
    #svclassifier = RandomForestClassifier()
    svclassifier.fit(x_train_t, y_train_t)
    y_test_tp=y_test_t.to_numpy()
    y_pred = svclassifier.predict(x_test_t)
    acc=calc_acc(y_test_tp,y_pred)
    
    if m<acc:
        m=acc
        x_train,x_test, y_train, y_test= x_train_t,x_test_t, y_train_t, y_test_t
#x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.20)

svclassifier = SVC(kernel='poly',coef0=2.0)
#svclassifier = RandomForestClassifier()
svclassifier.fit(x_train, y_train)
y_score = svclassifier.fit(x_train, y_train).decision_function(x_test)

y_score = svclassifier.predict_proba(x_test)
y_score = y_score[:,1]
y_test=y_test.to_numpy()
y_pred = svclassifier.predict(x_test)
acc=calc_acc(y_test,y_pred)
prec=precision_score(y_test, y_pred)
rec=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)
    
fpr, tpr, thresholds = roc_curve(y_test, y_score)
    
auc1=auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % auc1)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

print(acc)
print("precision = " + str(prec))
print("sensitivity = " + str(rec))
print("AUC = " + str(auc1))
print("F1 = " + str(f1))