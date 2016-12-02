
# coding: utf-8

# # Exercise 8 Part b

# ## Preparation

# #### Imports

# In[80]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### Import Datasets, split labels/ features, concatenate datasets

# In[81]:

red_wine = pd.read_csv('winequality-red.csv',sep=";")
white_wine = pd.read_csv('winequality-white.csv',sep=";")


# In[82]:

red_wine_labels = red_wine.ix[:,11]
red_wine_features = red_wine.ix[:,0:11]
white_wine_labels = white_wine.ix[:,11]
white_wine_features = white_wine.ix[:,0:11]


# In[83]:

both_wines = pd.concat([red_wine_features,white_wine_features],axis=0)
both_wines_labels = pd.concat([red_wine_labels,white_wine_labels],axis=0)


# In[25]:

both_wines.head()


# #### From pandas DataFrame to matrix (to work in sklearn)

# In[84]:

both_wines_array = both_wines.as_matrix()


# In[85]:

print(both_wines_array)


# ## 1. Clustering algorithm: KMeans

# #### Imports

# In[51]:

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


# #### List of KMeans clustering with different number of clusters to form

# In[29]:

kmeans = {'kmeans_2_20': KMeans(n_clusters=2,init='k-means++',n_init=20,random_state=5),
          'kmeans_3_20': KMeans(n_clusters=3,init='k-means++',n_init=20,random_state=5),
          'kmeans_4_20': KMeans(n_clusters=4,init='k-means++',n_init=20,random_state=5),
          'kmeans_5_20': KMeans(n_clusters=5,init='k-means++',n_init=20,random_state=5),
          'kmeans_6_20': KMeans(n_clusters=6,init='k-means++',n_init=20,random_state=5)
         }


# In[108]:

fignum = 1
results = {}
for k, v in sorted(kmeans.iteritems()):
    fig = plt.figure(fignum,figsize=(5,3))
    plt.clf()
    ax = Axes3D(fig,rect=[1,2,2,3],elev=10,azim=10)
    plt.cla()
    
    v.fit(both_wines_array)
    labels = v.labels_
    centroids = v.cluster_centers_
    
    ax.scatter(both_wines_array[:,3],both_wines_array[:,5],both_wines_array[:,6],c=labels.astype(np.float))
    plt.title(k)
    
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('residual sugar')
    ax.set_ylabel('free sulfur dioxide')
    ax.set_zlabel('total sulfur dioxide')
    
    from sklearn import metrics
    from sklearn.metrics import pairwise_distances

    sil = metrics.silhouette_score(both_wines_array,labels,metric='euclidean')
    results[k]=sil
    plt.show()
    print("Cluster:\n{}\n\n".format(k) + "Cluster memberships:\n{}\n\n".format(labels) + "Cluster Centers:\n{}\n\n".format(centroids) + "Silhouette Score:\n{}\n\n\n\n".format(sil))
    plt.close()
    fignum += 1


# ## 2. Clustering algorithm: AgglomerativeClustering

# In[86]:

from sklearn.cluster import AgglomerativeClustering


# In[91]:

aggloms = {'agglomerative_2_eucl': AgglomerativeClustering(n_clusters=2),
           'agglomerative_3_eucl': AgglomerativeClustering(n_clusters=3),
           'agglomerative_4_eucl': AgglomerativeClustering(n_clusters=4),
           'agglomerative_5_eucl': AgglomerativeClustering(n_clusters=5),
           'agglomerative_6_eucl': AgglomerativeClustering(n_clusters=6)
          }


# In[110]:

fignum = 1
for k, v in sorted(aggloms.iteritems()):
    fig = plt.figure(fignum,figsize=(5,3))
    plt.clf()
    ax = Axes3D(fig,rect=[1,2,2,3],elev=10,azim=130)
    plt.cla()
    
    v.fit(both_wines_array)
    labels = v.labels_
    
    ax.scatter(both_wines_array[:,3],both_wines_array[:,5],both_wines_array[:,6],c=labels.astype(np.float))
    plt.title(k)
    
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('residual sugar')
    ax.set_ylabel('free sulfur dioxide')
    ax.set_zlabel('total sulfur dioxide')
    
    from sklearn import metrics
    from sklearn.metrics import pairwise_distances

    sil = metrics.silhouette_score(both_wines_array,labels,metric='euclidean')
    results[k]=sil
    plt.show()
    print("Cluster:\n{}\n\n".format(k) + "Cluster memberships:\n{}\n\n".format(labels) + "Cluster Centers:\n{}\n\n".format(centroids) + "Silhouette Score:\n{}\n\n\n\n".format(sil))
    plt.close()
    
    fignum += 1


# ### Results

# In[133]:

import operator
sorted_results = sorted(results.items(), key = operator.itemgetter(1),reverse=True)
print "{:<30} {:<5}\n".format('Clustering Algorithm','Silhouette Score')

for each in sorted_results:
    print "{:<30} {:<5}".format(each[0],each[1])

