#!/usr/bin/env python
# coding: utf-8

# ## GRADUATE ROTATIONAL INTERNSHIP PROGRAMME (GRIP) - THE SPARKS FOUNDATION

# ##### DATA SCIENCE AND BUSINESS ANALYTICS - TASK 6 - PREDICTION USING DECISION TREE ALGORITHM
# _______________________________________________________________________________________________________
# ###### LEVEL - INTERMEDIATE

# ##### AUTHOR : SHRUTHI G
# _______________________________________________________________________________________________________________________________
# 
# 
# ##### TASK 6 - Create the Decision Tree Classifier and visualize it graphically.
# Dataset: https://bit.ly/3kXTdox (IRIS DATASET)

# ###### IMPORTING ALL THE NECESSARY PACKAGES

# In[29]:


import pandas as pd
import seaborn as sns
from sklearn import tree 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from jupyterthemes import jtplot
from sklearn.tree import export_graphviz

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer,f1_score,recall_score,precision_score

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# ###### LOAD AND CREATE A DATAFRAME FOR THE DATASET

# In[30]:


a=load_iris()
df=pd.DataFrame(a.data, columns=a.feature_names)
print(a.feature_names)
print("\n")
df['target']= a.target


# ###### PERFORMING SOME BASIC EDA

# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.shape


# In[31]:


corr = df.corr()
plt.figure(figsize=(5,5)) 
sns.heatmap(corr, annot=True)


# ###### SPLITTING THE DATASET INTO TEST AND TRAIN 

# In[9]:


X_train,X_test,Y_train,Y_test=train_test_split(df[a.feature_names],df['target'])
print(len(X_train))
print(len(X_test))


# ###### CREATING AN OBJECT FOR DECISION TREE CLASSIFIER

# In[10]:


DT=DecisionTreeClassifier(criterion="gini")


# ###### FIT AND PREDICT THE MODEL

# In[11]:


model=DT.fit(X_train,Y_train)
y_preds=model.predict(X_test)
print(y_preds)
print(a.feature_names)
print("\n")


# ###### PLOTTING THE DECISION TREE CLASSIFIER GRAPHICALLY

# In[43]:


axes=plt.subplots(nrows=1,ncols=1,figsize=(10,11),dpi=100)
tree.plot_tree(DT,fontsize=7,feature_names=a.feature_names,class_names=a.target_names)


# ###### CONFUSION MATRIX

# In[13]:


labels=[0,1,2]
cmx=confusion_matrix(Y_test,y_preds,labels)
print("\n Confustion Matrix : \n",cmx)
print("\n")
print(classification_report(Y_test,y_preds))

