
# -*- coding: utf-8 -*-

#Import libaries and functions needed
import pandas as pd 
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,RobustScaler 
from sklearn.linear_model import LogisticRegression,LinearRegression
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

import classify as cl
import os

#Import CSV Dataset
current_dir = os.getcwd()
data = pd.read_csv(os.path.join(current_dir,'bank-additional.csv'))

#EXPLORATORY DATA ANALYSIS
#Find the size of data
r,c = data.shape
# print(data.shape)

#Find the numerical featurs
print(data.describe())

#Data Imputing : Replace the unknown values with NaN
df = data.replace(to_replace = ['unknown'], value = np.NaN , regex = True)
#Find the columns that have NaN values
cols = df.columns[df.isna().any()].tolist()
print('Cols with unknown values :',cols)

#Histogram - Data Visualization 
plt.figure()
df.hist(figsize=(15,15),grid=False)

# Correlation MAtrix
plt.figure()
corr = df.corr()
sns.heatmap(corr,cmap = 'copper',annot=True,xticklabels=corr.columns.values,yticklabels=corr.columns.values)

#Box plots
plt.figure()
#Select the colums with numerical values 
df_box = df.ix[:,['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed']]
for i in range(1,10):
    plt.subplot(3,3,i)
    d = df_box.iloc[:,(i-1)]
    d.plot.box(sym='r+')
    
#Scatter Matrix
plt.figure()
scatter_matrix(df, alpha=0.2, figsize=(6, 6), diagonal='kde')

##Preprocessing of data 

#Removal of Missing values
# Find the percentage of missing values for each feature 
#number of missing values in each column 
#col_mval contains the number of missing elements in each column
col_mval = df.isna().sum().tolist()
#print(col_mval)
prob=np.zeros(np.size(col_mval))

#Probabiity of missing values 
for i in range(0,np.size(col_mval)):
    prob[i] = col_mval[i]/r
#If the probability is > 0.3, this means more than 30% of the values are missing then, remove that feature.

#Now remove only the missing instances 
   
#df = df.dropna()
#Check if there are no NaN values in the dataframe now 
#print(df1.isna().sum().tolist())
#print(df1.shape)            
#data without class labels is stored in X1
X1 = df.drop(df[df.education == 'illiterate'].index)
X1.drop(X1[X1.default == 'yes'].index,inplace = True)


#Impute Values Here .....

#Method 1 : Remove all the rows that contain NaN
#X1 = X1.dropna(axis = 0, how = 'any')

#Extract the class labels : cl_labels contains the numeric class labels 0/1
Y = X1.y.eq('yes').mul(1)
X1.drop(columns = 'y',inplace = True,axis = 1)
#X1.drop(columns = 'default',inplace = True,axis = 1)

sm = SMOTE(random_state=12, ratio = 1.0)
x_res, y_res = sm.fit_sample(X1, Y)
#Split into training and test data sets 
X_train1, X_test1, y_train1, y_test1 = train_test_split(x_res,y_res, shuffle = True, test_size=0.2,stratify = Y )
X_train1.sort_index(inplace = True)
X_tr1 = X_train1

#Replace the non - numerical columns with the classwise mode of the data
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

#Do one hot encoding for the entire data 
one_hot = pd.get_dummies(X_tr1)
#Drop the redundant values 
X_tr = one_hot
X_tr.drop(inplace = True,columns = ['housing_yes','loan_yes','contact_cellular'])

#Do one hot encoding for test values as well
one_hot = pd.get_dummies(X_test1)
X_ts = one_hot
X_ts.drop(inplace = True,columns = ['housing_yes','loan_yes','contact_cellular'])


#FEATURE SCALING
#scl = MinMaxScaler()
##scal = RobustScaler
#X_train = scl.fit_transform(X_tr,y_train1)
#X_test = scl.transform(X_ts)


#FEATURE SELECTION

full_X_oh = pd.get_dummies(X1)
full_X_oh.drop(inplace = True,columns = ['housing_yes','loan_yes','contact_cellular'])

#Method 1 - Using Logistic Regression
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



# Method2 - ExtraTrees Classifier
#from sklearn.ensemble import ExtraTreesClassifier
#forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
#forest.fit(full_X_oh, Y)
#coeff_feature = forest.feature_importances_ 

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

#Retain the first 'top_n_feat' number of features 
top_n_feat = 20
X_tr_r_list = (imp_feature_p.iloc[0:top_n_feat])
X_tr_r_list = list(X_tr_r_list.index)
#print(list(X_tr_r_list.index))
X_tr = X_tr[X_tr_r_list]
X_ts = X_ts[X_tr_r_list]


#Method 3 - KBEST
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


##NOrmalize each column using minmax scaler and apply to test
scl = MinMaxScaler()
scl.fit(X_tr,y_train1)
X_tr_n = (scl.transform(X_tr))
X_ts_n = (scl.transform(X_ts))


#PCA
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


#CLASSIFIER SELECTION ###Define Classifier
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'SVM - Linear',n_folds = 5)
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'SVM - RBF',n_folds = 5)
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'SVM - Polynomial',n_folds = 5)   
cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'Naive Bayes',n_folds = 5)
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'K Nearest Neighbors',n_folds = 5) 
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'MLP Classifier',n_folds = 5) 
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'RF Classifier',n_folds = 5)
#cl.classifier_models(X_tr2,X_ts2,y_tr2,y_ts2,'Decision Tree Classifier',n_folds = 5)
#cl.classifier_models(X_train,X_test,y_tr2,y_ts2,'SVM - Linear',n_folds = 5)   
