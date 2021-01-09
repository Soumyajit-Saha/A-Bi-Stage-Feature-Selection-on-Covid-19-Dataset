# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 17:56:13 2020

@author: Soumyajit Saha
"""


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
from sklearn.ensemble import RandomForestClassifier

###### Calculation of accuracy #####
def calc_acc(y, y_pred):
    total=len(y)
    c=0
    for i in range(0,total):
        if y[i]==y_pred[i]:
            c+=1
    return c/total

####### Calculation of fitness #########
def fitness(moth_pos, x_train, y_train, x_test, y_test):
    
    x_train_selected_features=pd.DataFrame()
    x_test_selected_features=pd.DataFrame()


    for i in range(len(moth_pos)):
        x_train_selected_features[str(moth_pos[i])] = x_train.iloc[:,int(moth_pos[i])]
        x_test_selected_features[str(moth_pos[i])] = x_test.iloc[:,int(moth_pos[i])]
    
    svclassifier = SVC(kernel='poly', coef0=2.0)
    #svclassifier = RandomForestClassifier()
    svclassifier.fit(x_train_selected_features, y_train)
    y_score = svclassifier.fit(x_train_selected_features, y_train).decision_function(x_test_selected_features)
    #y_score = svclassifier.predict_proba(x_test_selected_features)
    y_score = y_score[:,1]
    
    y_test=y_test.to_numpy()
    y_pred = svclassifier.predict(x_test_selected_features)

    acc=calc_acc(y_test,y_pred)
    prec=precision_score(y_test, y_pred)
    rec=recall_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    
    auc1=auc(fpr, tpr)
    
    return acc, prec, rec, f1, auc1, fpr, tpr

def distance(a,b,dim):
    o = np.zeros(dim)
    for i in range(0,len(a)):
        o[i] = abs(a[i] - b[i])
    return o

def Levy(d):
    beta=3/2
    sigma=(math.gamma(1+beta)*math.sin(math.pi*beta/2)/(math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
    u=np.random.randn(d)*sigma
    v=np.random.randn(d)
    step=u/abs(v)**(1/beta)
    o=0.01*step
    return o     

def DA(x_train, y_train, x_test, y_test, Max_iteration, m, orig):
    
    length=int(len(x_train.iloc[0]))
    
    dim=length
    
    ub=int(len(x_train.iloc[0]))-1
    lb=0
    
    r=(ub-lb)/10
    Delta_max=(ub-lb)/8.5
    
    Food_fitness=m
    Food_pos=orig
    
    Enemy_fitness=math.inf
    Enemy_pos=np.zeros(dim)
    
    fitness_of_X = np.zeros(20)
    All_fitness = np.zeros(20)
    
    X = np.zeros(shape=(20,dim))
    DeltaX = np.zeros(shape=(20,dim))
    
    for i in range(0,20):
        X[i]=np.random.randint(len(x_train.iloc[0]), size=length)
        # for j in range(0,dim):
        #     X[i][j]=int(lb + random.uniform(0,1)*(ub-lb))
        
        X[i] = np.sort(X[i])
    
    i1=random.randint(0,19)
    i2=random.randint(0,19)
    
    while i2==i1:
        i2=random.randint(0,19)
        
    
    ub_del=max(distance(X[i1],X[i2],dim))
    
    
    
    for i in range(0,20):
        for j in range(0,dim):
            DeltaX[i][j]=int(lb + random.uniform(0,1)*(ub_del-lb))

    for itr in range(1,Max_iteration+1):
        
        r=(ub_del-lb)/4+((ub_del-lb)*(itr/Max_iteration)*2)
        w=0.9-itr*((0.9-0.4)/Max_iteration)
        my_c=0.1-itr*((0.1-0)/(Max_iteration/2))
        
        if my_c<0:
            my_c=0
        
        s=2*random.random()*my_c
        a=2*random.random()*my_c
        c=2*random.random()*my_c
        f=2*random.random()*my_c
        e=my_c
        
        for i in range(0,20):
            fitness_of_X[i], precision, sensitivity, F1, AUC, fpr, tpr = fitness(X[i],x_train, y_train, x_test, y_test)
            All_fitness[i] = fitness_of_X[i]
            
            if fitness_of_X[i] > Food_fitness:
                Food_fitness = fitness_of_X[i]
                Food_precision=precision
                Food_sensitivity=sensitivity
                Food_F1=F1
                Food_AUC=AUC
                Food_pos=X[i]
                Food_fpr=fpr
                Food_tpr=tpr
            
        
            
            if fitness_of_X[i] < Enemy_fitness:
                if all((X[i] <= ub)) and all((X[i] >= lb)):
                    Enemy_fitness = fitness_of_X[i]
                    Enemy_pos = X[i]
        
        print(Food_fitness)
        print(len(np.unique(Food_pos)))
        for i in range(0,20):
            index=0
            neighbours_no=0
            
            Neighbours_X = np.zeros(shape=(20,dim))
            Neighbours_DeltaX = np.zeros(shape=(20,dim))
            
            for j in range(0,20):
                Dist2Enemy = distance(X[i],X[j],dim)
                if (all(Dist2Enemy<=r) and all(Dist2Enemy!=0)):
                    index=index+1
                    neighbours_no=neighbours_no+1
                    Neighbours_DeltaX[index]=DeltaX[j]
                    Neighbours_X[index]=X[j]
                    
            S=np.zeros(dim)           
            if neighbours_no>1:
                for k in range(0,neighbours_no):
                    S=S+(Neighbours_X[k]-X[i])
                S=-S
            else:
                S=np.zeros(dim)
                
            
            
            if neighbours_no>1:
                A=(sum(Neighbours_DeltaX))/neighbours_no
            else:
                A = DeltaX[i]
            
            
            
            if neighbours_no>1:
                C_temp=(sum(Neighbours_X))/neighbours_no
            else:
                C_temp=X[i]
        
            C=C_temp-X[i]
            
            
            
            Dist2Food=distance(X[i],Food_pos,dim)
                               
            if all(Dist2Food<=r):
                F=Food_pos-X[i]
            else:
                F=np.zeros(dim)
            
            
            
            Dist2Enemy=distance(X[i],Enemy_pos,dim)
                               
            if all(Dist2Enemy<=r):
                Enemy=Enemy_pos-X[i]
            else:
                Enemy=np.zeros(dim)
            
            
            
            for tt in range(0,dim):
                if X[i][tt]>ub:
                    X[i][tt]=ub
                    DeltaX[i][tt]=random.uniform(0,1)*(50-lb)
                    
                if X[i][tt]<lb:
                    X[i][tt]=lb
                    DeltaX[i][tt]=random.uniform(0,1)*(50-lb)
            
            temp=np.zeros(dim)
            Delta_temp=np.zeros(dim)
            
            if any(Dist2Food>r):
                if neighbours_no>1:
                    for j in range(0,dim):                                               
                        Delta_temp[j] = int(w*DeltaX[i][j] + random.random()*A[j] + random.random()*C[j] + random.random()*S[j])
                        if Delta_temp[j]>Delta_max:
                            Delta_temp[j]=Delta_max
                        if Delta_temp[j]<-Delta_max:
                            Delta_temp[j]=-Delta_max
                        temp[j]=X[i][j]+(Delta_temp[j])
                else:
                    temp=(X[i] + (Levy(dim))*X[i]).astype(int)
                    Delta_temp=np.zeros(dim)
            
            else:
                for j in range(0,dim):
                    Delta_temp[j] = int((a*A[j] + c*C[j] + s*S[j] + f*F[j] + e*Enemy[j]) + w*DeltaX[i][j])
                    if Delta_temp[j]>Delta_max:
                        Delta_temp[j]=Delta_max
                    if Delta_temp[j]<-Delta_max:
                        Delta_temp[j]=-Delta_max
                    temp[j]=X[i][j]+Delta_temp[j]
                    
            for j in range(0,dim):
                if temp[j]<lb: # Bringinging back to search space
                        temp[j]=lb
                    
                if temp[j]>ub: # Bringinging back to search space
                    temp[j]=ub
            acc, precision1, sensitivity1, F1_1, AUC1, fpr1, tpr1= fitness(temp,x_train, y_train, x_test, y_test)       
            #if acc > fitness_of_X[i]:
            if(fitness(temp,x_train, y_train, x_test, y_test)) > Food_fitness:
                X[i]=temp
                DeltaX[i]=Delta_temp
            
            
                    
        Best_score=Food_fitness
        Best_pos=Food_pos
        
        print("Iteration = " + str(itr))
    
    plt.figure()
    lw = 2
    plt.plot(Food_fpr, Food_tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % Food_AUC)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return Best_pos, Best_score, Food_precision, Food_sensitivity, Food_F1, Food_AUC    