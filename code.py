
# -*- coding: utf-8 -*-

#Import libaries and functions needed
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,RobustScaler 
from sklearn.linear_model import LogisticRegression,LinearRegression
import matplotlib.pyplot as plt
import classify as cl
import os

#Import CSV Dataset
current_dir = os.getcwd()
data = pd.read_csv(os.path.join(current_dir,'bank-additional.csv'))

#EXPLORATORY DATA ANALYSIS
#Find the size of data
r,c = data.shape
print(data.shape)

#Find teh numerical featurs
print(data.describe())

#Replace the unknown values with NaN
df = data.replace(to_replace = ['unknown'], value = np.NaN , regex = True)
print('Cols with unknown values :')
#Find the columns that have NaN values
cols = df.columns[df.isna().any()].tolist()
print(cols)

##################Data Visualization ########################
#Histogram - Data Visualization 
plt.figure()
df.hist(figsize=(15,15),grid=False)

# COrrelation MAtrix
plt.figure()
import seaborn as sns
corr = df.corr()
sns.heatmap(corr,cmap = 'copper',annot=True,xticklabels=corr.columns.values,yticklabels=corr.columns.values)

#Box plots
plt.figure()
#Select teh colums with numerical values 
df_box = df.ix[:,['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]
for i in range(1,10):
    plt.subplot(3,3,i)
    d = df_box.iloc[:,(i-1)]
    d.plot.box(sym='r+')
    
#Scatter Matrix
plt.figure()
from pandas.plotting import scatter_matrix
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

############### Preprocessing ####################

#******************Removal of Missing values********************#
# Find the percentage of missing values for each feature 

#number of missing values in each column 
#col_mval contains the number of missing elements in each column
col_mval = df.isna().sum().tolist()
#print(col_mval)
prob=np.zeros(np.size(col_mval))

#Probabiity of missing values 
for i in range(0,np.size(col_mval)):
    prob[i] = col_mval[i]/r
#If the probability is > 0.3, this means more than 30% of the values are missing,
#then, remove that feature.
#NOw remove only the missing instances 
    
#df = df.dropna()

#Check if there are no NaN values in the dataframe now 
#print(df1.isna().sum().tolist())
#print(df1.shape)            
#****************************         
#data without class labels is stored in X1
X1 = df.drop(df[df.education == 'illiterate'].index)
X1.drop(X1[X1.default == 'yes'].index,inplace = True)


#Impute Values Here .....

#*****Method 1 : Remove all the rows that contain NaN*******
#X1 = X1.dropna(axis = 0, how = 'any')
#

#Extract the class labels : cl_labels contains the numeric class labels 0/1
Y = X1.y.eq('yes').mul(1)
X1.drop(columns = 'y',inplace = True,axis = 1)
#X1.drop(columns = 'default',inplace = True,axis = 1)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=12, ratio = 1.0)
x_res, y_res = sm.fit_sample(X1, Y)
#Split into training and test data sets 
X_train1, X_test1, y_train1, y_test1 = train_test_split(x_res,y_res, shuffle = True, test_size=0.2,stratify = Y )
X_train1.sort_index(inplace = True)


X_tr1 = X_train1
#******************Replace the non - numerical columns with the classwise mode of the data
#Find the columns that have NaN values
#Extract the column data for the columns with missing values 
data_c1 = X_train1.loc[y_train1 == 0]
cols = data_c1.columns[data_c1.isna().any()].tolist()
data_c1[cols]=data_c1[cols].fillna(data_c1.mode().iloc[0])

data_c2 = X_train1.loc[y_train1 == 1]
cols = data_c2.columns[data_c2.isna().any()].tolist()
data_c2[cols]=data_c2[cols].fillna(data_c2.mode().iloc[0])
#print(data_c2.shape)

#df1 contains the missing data replaced with classwise mode 
X_tr1 =pd.concat([data_c1,data_c2])

#***************************************************************************************

#Do one hot encoding for the entire data 
one_hot = pd.get_dummies(X_tr1)
#Drop the redundant values 
X_tr = one_hot
X_tr.drop(inplace = True,columns = ['housing_yes','loan_yes','contact_cellular'])

#Do one hot encoding for test values as well
one_hot = pd.get_dummies(X_test1)
X_ts = one_hot
X_ts.drop(inplace = True,columns = ['housing_yes','loan_yes','contact_cellular'])


############Feature Scaling##########
#
#scl = MinMaxScaler()
##scal = RobustScaler
#X_train = scl.fit_transform(X_tr,y_train1)
#X_test = scl.transform(X_ts)


####################Feature Selection ########################

full_X_oh = pd.get_dummies(X1)
full_X_oh.drop(inplace = True,columns = ['housing_yes','loan_yes','contact_cellular'])

########## Method 1 - Using Logistic Regression
plt.figure()
lr_feature = LogisticRegression()
lr_feature.fit(full_X_oh,Y)
coeff_feature = lr_feature.coef_

imp_feature = pd.Series(coeff_feature[0],index=list(full_X_oh))
#Feature on both axes are important 
imp_feature_p = imp_feature.sort_values(ascending = False)
imp_feature_p.plot.barh()
plt.title('Feature Selection by Logistic Regression')
plt.show()



###################Feature Sel - Method2 - ExtraTrees Classifier
#from sklearn.ensemble import ExtraTreesClassifier
#forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
#forest.fit(full_X_oh, Y)
#coeff_feature = forest.feature_importances_ 
# 
#
#imp_feature = pd.Series(coeff_feature[0],index=list(full_X_oh))
##Feature on both axes are important 
##imp_feature_p = imp_feature.sort_values(ascending = False)
##imp_feature_p.plot.barh()
##plt.show()
#indices = np.argsort(coeff_feature)[::-1]
#plt.title("Feature importances")
#plt.bar(range(full_X_oh.shape[1]), coeff_feature[indices],color="r", align="center")
#plt.xticks(range(full_X_oh.shape[1]), indices)
#plt.xlim([-1, full_X_oh.shape[1]])
#plt.show()

##########################################################################
#Retain the first 'top_n_feat' number of features 
top_n_feat = 20
X_tr_r_list = (imp_feature_p.iloc[0:top_n_feat])
X_tr_r_list = list(X_tr_r_list.index)
#print(list(X_tr_r_list.index))

X_tr = X_tr[X_tr_r_list]
X_ts = X_ts[X_tr_r_list]
#


###############Method 3 - KBEST
#from sklearn.feature_selection import SelectKBest
#np.seterr(divide='ignore', invalid='ignore')
#kbest = SelectKBest(k=5)
#
#kbest.fit_transform(X_tr,y_train1)
#X_ts2 = kbest.transform(X_ts)
#
#coeff_feature = kbest.scores_
#
#imp_feature = pd.Series(coeff_feature[0],index=list(full_X_oh))
##Feature on both axes are important 
##imp_feature_p = imp_feature.sort_values(ascending = False)
#coeff_feature.plot.barh()
#plt.title('Feature Selection by SelectKBest')
#plt.show()

#########################################
##NOrmalize each column using minmax scaler and apply to test
scl = MinMaxScaler()
scl.fit(X_tr,y_train1)
X_tr_n = (scl.transform(X_tr))
X_ts_n = (scl.transform(X_ts))

###################PCA
#from sklearn.decomposition import PCA
#pca = PCA(n_components=15)
#pca.fit(X_tr)
#X_tr = pca.transform(X_tr)
#X_ts = pca.transform(X_ts)

 
X_tr2 = pd.DataFrame.as_matrix(X_tr)
X_ts2 =  pd.DataFrame.as_matrix(X_ts)
#X_tr2 = X_train
#X_ts2 = X_test
y_tr2 = pd.DataFrame.as_matrix(y_train1)
y_ts2 = pd.DataFrame.as_matrix(y_test1)

#############CLASSIFIER SELECTION ###################
##Define Classifier
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'SVM - Linear',n_folds = 5)
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'SVM - RBF',n_folds = 5)
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'SVM - Polynomial',n_folds = 5)   
cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'Naive Bayes',n_folds = 5)
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'K Nearest Neighbors',n_folds = 5) 
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'MLP Classifier',n_folds = 5) 
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'RF Classifier',n_folds = 5)
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'Decision Tree Classifier',n_folds = 5)
#cl.classifier_models(X_train,X_test,y_tr2,y_ts2,'SVM - Linear',n_folds = 5)   

__________________________________________________________________________________________________


# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 21:28:35 2018

Created on Wed Apr 18 01:40:08 2018

EE559 - MPR - Final Project 
Development of a Pattern Recognition System
@author: Muthulakshmi Chandrasekaran
muthulac@usc.edu
Submission Date : May 1, 2018
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
#    from sklearn.cross_validation import cross_val_predict
    roc = metrics.roc_auc_score(y_ts, y_pred_proba)
    print('AUC_ROC Score  :',roc)
    
      
    plt.plot([0,1],[0,1],color='red',lw=2,linestyle='--')
    plt.plot(fpr,tpr,label="data 1, roc="+str(roc))
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.show()
    
    
    
    

#FInd best parameters for Linear SVM using cross validation 
    
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

    ______________________________________________________________________________________________
    
    
    

    






