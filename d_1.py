
# coding: utf-8

# In[89]:

from __future__ import print_function

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, recall_score, precision_score


# In[90]:

red_wine = pd.read_csv('winequality-red.csv',sep=";")
red_wine['type']="red wine"
white_wine = pd.read_csv('winequality-white.csv',sep=";")
white_wine['type']="white wine"


# In[91]:

def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)


# #### From two Datasets to one
# #### Factorize type of wine (red wine --> 0, white wine --> 1)

# In[92]:

from sklearn.model_selection import train_test_split

both_wines = pd.concat([red_wine,white_wine],axis=0)

df2, targets = encode_target(both_wines, "type")
print("* df2.head()", df2[["Target", "type"]].head(),
      sep="\n", end="\n\n")
print("* df2.tail()", df2[["Target", "type"]].tail(),
      sep="\n", end="\n\n")
print("* targets", targets, sep="\n", end="\n\n")


# #### Shuffle the dataset

# In[93]:

df_shuffled = df2.iloc[np.random.permutation(len(df2))]
df2 = df_shuffled.reset_index(drop=True)


# In[94]:

from sklearn.preprocessing import StandardScaler
features = list(df2.columns[:12])
print("* features:", features, sep="\n")

y = df2["Target"]
X = df2[features]
X_std = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std,y,test_size=.3,random_state=10)


# In[95]:

def con_rec_pre(y_true,y_pred):
    """Calculates and prints the confusion matrix, the recall and the precision score

    Args
    ----
    y_true -- 1d array-like, or label indicator array / sparse matrix
    y_pred -- 1d array-like, or label indicator array / sparse matrix

    Returns
    -------
    test_conf_mat -- confusion matrix
    test_recall -- recall score
    test_precision -- precision score
    """
    test_conf_mat = confusion_matrix(y_true,y_pred)
    test_recall = recall_score(y_true,y_pred)
    test_precision = precision_score(y_true,y_pred)
    print("* Confusion matrix:", test_conf_mat,sep="\n", end="\n\n")
    print("* Recall:", test_recall,sep="\n", end="\n\n")
    print("* Precision:", test_precision,sep="\n", end="\n\n")


# ### 1. Classifier Algorithm: Decision Trees

# In[96]:

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=10)
dt.fit(X_train,y_train)

dt_y_train_pred = dt.predict(X_train)
dt_y_test_pred = dt.predict(X_test)


# In[97]:

con_rec_pre(dt_y_test_pred,y_test)


# In[98]:

con_rec_pre(dt_y_train_pred,y_train)


# In[99]:

from sklearn.tree import export_graphviz
with open("dt.dot", 'w') as f:
    export_graphviz(dt, out_file=f,feature_names=features)


# ### 2. Classifier Algorithm: KNN

# In[100]:

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train,y_train)

knn_y_train_pred = knn.predict(X_train)
knn_y_test_pred = knn.predict(X_test)


# In[101]:

con_rec_pre(knn_y_train_pred,y_train)


# In[102]:

con_rec_pre(knn_y_test_pred,y_test)


# ### 3. Classifier Algorithm: Naive Bayes

# In[103]:

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train,y_train)

nb_y_train_pred = nb.predict(X_train)
nb_y_test_pred = nb.predict(X_test)


# In[104]:

con_rec_pre(nb_y_train_pred,y_train)


# In[105]:

con_rec_pre(nb_y_test_pred,y_test)

