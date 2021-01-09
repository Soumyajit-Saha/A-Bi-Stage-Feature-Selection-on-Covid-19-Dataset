# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 21:13:10 2020

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
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import random
import math 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt




def calc_acc(y, y_pred):
    total=len(y)
    c=0
    for i in range(0,total):
        if y[i]==y_pred[i]:
            c+=1
    return c,total,c/total

dataset=pd.read_csv('D:/Project/COVID-19/reduced_feature_set2.csv') 
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

x_train,x_test, y_train, y_test=train_test_split(x,y,test_size=0.20)
# svclassifier = SVC(kernel='linear',coef0=0.0)
# y_score = svclassifier.fit(x_train, y_train).decision_function(x_test)
svclassifier = RandomForestClassifier(n_estimators=300)
svclassifier.fit(x_train, y_train)
y_score = svclassifier.predict_proba(x_test)
y_score = y_score[:,1]
    
y_test=y_test.to_numpy()
y_pred = svclassifier.predict(x_test)

c,t,acc=calc_acc(y_test,y_pred)
prec=precision_score(y_test, y_pred)
rec=recall_score(y_test, y_pred)
f1=f1_score(y_test, y_pred)
    
fpr, tpr, thresholds = roc_curve(y_test, y_score)
    
auc1=auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc1)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, label='Reference line', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

print('accuracy='+str(acc))
print('precision='+str(prec))
print('recall='+str(rec))
print('F1='+str(f1))
print('AUC='+str(auc1))
print(c)
print(t)