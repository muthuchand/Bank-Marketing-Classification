
# -*- coding: utf-8 -*-
"""
Implementation of all Classifiers
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn import metrics 
import numpy as np


#DIfferent classifier Models are defined 
def classifier_models(X_tr,X_ts,y_tr,y_ts,name,n_folds) :
    
    #Define  different Classifiers
    #Linear SVM
    if name == 'SVM - Linear' :
        print('--- SVM - Linear --- ')
        opt_gamma, opt_c = svm_linear_getparam(X_tr,X_ts,y_tr,y_ts,folds = n_folds)
        opt_model = SVC(kernel='linear',C = opt_c,gamma = opt_gamma,probability = True)
    
    # SVM - RBF
    elif name == 'SVM - RBF':
        print(' --- SVM - RBF --- ')
        opt_gamma, opt_c = svm_rbf_getparam(X_tr,X_ts,y_tr,y_ts,folds = n_folds)
        opt_model = SVC(kernel='rbf',C = opt_c,gamma = opt_gamma,probability = True)
        
    #SVM - Polynomial
    elif name == 'SVM - Polynomial':
        print(' --- SVM - Polynomial --- ')
        opt_gamma, opt_c,opt_deg = svm_poly_getparam(X_tr,X_ts,y_tr,y_ts,folds = n_folds)
        opt_model = SVC(kernel='poly',degree = opt_deg, C = opt_c,gamma = opt_gamma,probability = True)
        
    #SVM - Naive Bayes
    elif name == 'Naive Bayes':
        print(' --- Gaussian NB Classifier --- ')
        opt_model = GaussianNB()
        
    #KNN
    elif name == 'K Nearest Neighbors':
        print(' --- K Nearest Neighbors  --- ')
        opt_neigh = knn_getparam(X_tr,X_ts,y_tr,y_ts,folds = n_folds)
        opt_model = KNeighborsClassifier(n_neighbors = opt_neigh)
    
    #MLP
    elif name == 'MLP Classifier' :
        print(' --- Multi Layer Perceptron --- ')
        h1 = mlp_getparam(X_tr,X_ts,y_tr,y_ts,folds = n_folds)
        opt_model = MLPClassifier(hidden_layer_sizes=h1)
#        opt_model = MLPClassifier(max_iter = 400,batch_size = 200,alpha = 0.005,beta_1 = 0.1,beta_2 = 0.88,hidden_layer_sizes=(20,20))
   
    #RF Classifier
    elif name == 'RF Classifier' :
        print(' --- Random Forest Classifier --- ')
        opt_est = rf_getparam(X_tr,X_ts,y_tr,y_ts,folds = n_folds)
        opt_model = RandomForestClassifier(n_estimators = opt_est)
        
    # Decision Tree Classifier
    elif name == 'Decision Tree Classifier':
        print(' --- Decision Tree Classifier --- ')
        opt_depth = dt_getparam(X_tr,X_ts,y_tr,y_ts,folds = n_folds)
        opt_model = DecisionTreeClassifier(max_depth = opt_depth)

    #Train the model 
    opt_model.fit(X_tr,y_tr)
    y_pred = opt_model.predict(X_ts)
    
    #Accuracy Metrics
    acc = metrics.accuracy_score(y_ts,y_pred)
    prec, rec, f1_sc, support = metrics.precision_recall_fscore_support(y_ts, y_pred,average = 'weighted')
#    print(y_pred)
    roc = metrics.roc_auc_score(y_pred,y_ts)
    print('Test Accuracy : ',acc)
#    print('Classification Report :')
#    print(classification_report(y_ts, y_pred))
    print('Precision :',prec)
    print('Recall :',rec)

    print('F1 Score :',f1_sc)
    #PLot ROC
    plt.figure()
    y_pred_proba = opt_model.predict_proba(X_ts)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_ts,  y_pred_proba)
    roc = metrics.roc_auc_score(y_ts, y_pred_proba)
    print('AUC_ROC Score  :',roc)
    
      
    plt.plot([0,1],[0,1],color='red',lw=2,linestyle='--')
    plt.plot(fpr,tpr,label="data 1, roc="+str(roc))
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.show()
    

#Find best parameters for Linear SVM using cross validation 
    
def svm_linear_getparam(X_tr,X_ts,y_tr,y_ts,folds):
    skf = StratifiedKFold(n_splits = folds,shuffle = True)
    C_val = np.logspace(-3,3,num=3);
    gamma_val = np.logspace(-3,3,num=3);
    size_gamma = np.size(gamma_val)
    size_c = np.size(C_val)
#    max_sd = 1000
#    ind_i = 0
#    ind_j = 0
    cnt = 1    
    g = -10 
    c = -10
    acc_max = -100
    for i in range(0,size_gamma):
        for j in range(0,size_c):
#            print('Iteration : ',cnt);
            cnt=cnt+1;
            sample_gamma = gamma_val[i]
            sample_c = C_val[j]
            for tr_ind,v_ind in skf.split(X_tr,y_tr):
                X_train, X_val = X_tr[tr_ind],X_tr[v_ind]
                y_train, y_val = y_tr[tr_ind],y_tr[v_ind]
                model = SVC(kernel='linear',C = sample_c,gamma = sample_gamma)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_val)
                acc_val = metrics.accuracy_score(y_val,y_pred)
                if acc_val > acc_max:
                    acc_max = acc_val
                    g = gamma_val[i]
                    c = C_val[j]
    
    print("Training Accuracy : ",acc_max)
    print("Optimal Values : g : ",g," c:",c)
    return g,c


#FInd best parameters for RBF SVM using cross validation 
def svm_rbf_getparam(X_tr,X_ts,y_tr,y_ts,folds):
    skf = StratifiedKFold(n_splits = folds,shuffle = True)
    C_val = np.logspace(-3,3,num=3);
    gamma_val = np.logspace(-3,3,num=3);
    size_gamma = np.size(gamma_val)
    size_c = np.size(C_val)
#    max_sd = 1000
#    ind_i = 0
#    ind_j = 0
    cnt = 1    
    g = -10 
    c = -10
    acc_max = -100
    for i in range(0,size_gamma):
        for j in range(0,size_c):
#            print('Iteration : ',cnt);
            cnt=cnt+1;
            sample_gamma = gamma_val[i]
            sample_c = C_val[j]
            for tr_ind,v_ind in skf.split(X_tr,y_tr):
                X_train, X_val = X_tr[tr_ind],X_tr[v_ind]
                y_train, y_val = y_tr[tr_ind],y_tr[v_ind]
                model = SVC(kernel='rbf',C = sample_c,gamma = sample_gamma)
                model.fit(X_train,y_train)
                y_pred = model.predict(X_val)
                acc_val = metrics.accuracy_score(y_val,y_pred)
                if acc_val > acc_max:
                    acc_max = acc_val
                    g = gamma_val[i]
                    c = C_val[j]
    print("Training Accuracy : ",acc_max)
    print("Optimal Values : g : ",g," c:",c)
    return g,c


#FInd best parameters for polynomial SVM using cross validation 
def svm_poly_getparam(X_tr,X_ts,y_tr,y_ts,folds):
    skf = StratifiedKFold(n_splits = folds,shuffle = True)
    C_val = np.logspace(-3,3,num=5);
    gamma_val = np.logspace(-3,3,num=5);
    size_gamma = np.size(gamma_val)
    size_c = np.size(C_val)
#    max_sd = 1000
#    ind_i = 0
#    ind_j = 0
    cnt = 1    
    g = -10 
    c = -10
    deg = 0
    acc_max = -100
    for i in range(0,size_gamma):
        for j in range(0,size_c):
            for k in range(1,6):
#                print('Iteration : ',cnt);
                cnt=cnt+1;
                sample_gamma = gamma_val[i]
                sample_c = C_val[j]
                for tr_ind,v_ind in skf.split(X_tr,y_tr):
                    X_train, X_val = X_tr[tr_ind],X_tr[v_ind]
                    y_train, y_val = y_tr[tr_ind],y_tr[v_ind]
                    model = SVC(kernel='poly',degree = 3, C = sample_c,gamma = sample_gamma)
                    model.fit(X_train,y_train)
                    y_pred = model.predict(X_val)
                    acc_val = metrics.accuracy_score(y_val,y_pred)
                    if acc_val > acc_max:
                        acc_max = acc_val
                        g = gamma_val[i]
                        c = C_val[j]
                        deg = k
    print("Training Accuracy : ",acc_max)                    
    print("Optimal Values : g : ",g," c:",c,'degree :',deg)
    return g,c,deg


#FInd best parameters for KNN using cross validation 
def knn_getparam(X_tr,X_ts,y_tr,y_ts,folds):
    skf = StratifiedKFold(n_splits = folds,shuffle = True)
    g = np.arange(1,10)
    acc_max = -1000
    n_opt = -10
    for i in range(0,np.size(g)):
        n_trial = g[i]
        for tr_ind,v_ind in skf.split(X_tr,y_tr):
            X_train, X_val = X_tr[tr_ind],X_tr[v_ind]
            y_train, y_val = y_tr[tr_ind],y_tr[v_ind]
            model = KNeighborsClassifier(n_neighbors = n_trial)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_val)
            acc_val = metrics.accuracy_score(y_val,y_pred)
            if acc_val > acc_max:
                acc_max = acc_val
                n_opt = n_trial
                        
    print("Optimal Value : No.of neighbors : ",n_opt)
    print("Training Accuracy : ",acc_max)
    return n_opt
        

#FInd best parameters for MLP using cross validation 
def mlp_getparam(X_tr,X_ts,y_tr,y_ts,folds):
    skf = StratifiedKFold(n_splits = folds,shuffle = True)
    h1 = np.arange(1,200)
    acc_max = -1000
    h1_opt = -10
    for i in range(0,np.size(h1)):
        h11 = h1[i]
        for tr_ind,v_ind in skf.split(X_tr,y_tr):
            X_train, X_val = X_tr[tr_ind],X_tr[v_ind]
            y_train, y_val = y_tr[tr_ind],y_tr[v_ind]
            model = MLPClassifier(hidden_layer_sizes=h11)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_val)
            acc_val = metrics.accuracy_score(y_val,y_pred)
            if acc_val > acc_max:
                acc_max = acc_val
                h1_opt = h1
                         
    print("Training Accuracy : ",acc_max)                    
    print("Optimal Value : Hidden layer size : ",h1_opt)
    return h1_opt


#FInd best parameters for Random Forest using cross validation 
def rf_getparam(X_tr,X_ts,y_tr,y_ts,folds):
    skf = StratifiedKFold(n_splits = folds,shuffle = True)
    g = np.arange(1,100)
    acc_max = -1000
    opt_est = -10
    for i in range(0,np.size(g)):
        n_trial = g[i]
        for tr_ind,v_ind in skf.split(X_tr,y_tr):
            X_train, X_val = X_tr[tr_ind],X_tr[v_ind]
            y_train, y_val = y_tr[tr_ind],y_tr[v_ind]
            model = RandomForestClassifier(n_estimators = n_trial)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_val)
            acc_val = metrics.accuracy_score(y_val,y_pred)
            if acc_val > acc_max:
                acc_max = acc_val
                opt_est = n_trial
    print("Training Accuracy : ",acc_max)                    
    print("Optimal Value : No.of estimators : ",opt_est)
    return opt_est


#FInd best parameters for Decision Tree using cross validation 
def dt_getparam(X_tr,X_ts,y_tr,y_ts,folds):
    skf = StratifiedKFold(n_splits = folds,shuffle = True)
    g = np.arange(1,100)
    acc_max = -1000
    opt_dep = -10
    for i in range(0,np.size(g)):
        n_trial = g[i]
        for tr_ind,v_ind in skf.split(X_tr,y_tr):
            X_train, X_val = X_tr[tr_ind],X_tr[v_ind]
            y_train, y_val = y_tr[tr_ind],y_tr[v_ind]
            model = DecisionTreeClassifier(max_depth = n_trial)
            model.fit(X_train,y_train)
            y_pred = model.predict(X_val)
            acc_val = metrics.accuracy_score(y_val,y_pred)
            if acc_val > acc_max:
                acc_max = acc_val
                opt_dep = n_trial
    print("Training Accuracy : ",acc_max)                    
    print("Optimal Value : Max depth : ",opt_dep)
    return opt_dep

  






