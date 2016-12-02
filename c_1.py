
# coding: utf-8

# In[73]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[38]:

red_wine = pd.read_csv('winequality-red.csv',sep=";")
white_wine = pd.read_csv('winequality-white.csv',sep=";")


# In[39]:

red_wine_labels = red_wine.ix[:,11]
red_wine_features = red_wine.ix[:,0:11]
white_wine_labels = white_wine.ix[:,11]
white_wine_features = white_wine.ix[:,0:11]


# In[76]:

both_wines = pd.concat([red_wine_features,white_wine_features],axis=0)
both_wines_labels = pd.concat([red_wine_labels,white_wine_labels],axis=0)


# In[41]:

both_wines_array = both_wines.as_matrix()


# In[42]:

from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(both_wines)


# In[110]:

from pylab import pcolor, show, colorbar, xticks, yticks
print "Covariance matrix \n{}".format(np.cov(X_std.T))
fig = plt.figure()

fig.clf()
pcolor(np.cov(X_std.T),cmap=plt.cm.coolwarm)
colorbar(cmap=plt.cm.coolwarm)
yticks(np.arange(0.5,11.5),range(1,12))
xticks(np.arange(0.5,11.5),range(1,12))
plt.title("Covariance Matrix of Standardized")
plt.show()


# In[44]:

cov_mat = np.cov(X_std.T)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)

print "Eigenvectors:{}\n".format(eig_vecs)
print "\nEigenvalues:{}\n".format(eig_vals)


# In[45]:

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in  range(len(eig_vals))]

eig_pairs.sort()
eig_pairs.reverse()

print "Eigenvalues in descending order:"
for i in eig_pairs:
    print(i[0])


# In[46]:

from sklearn.decomposition import PCA

pca = PCA(n_components=11)
pca.fit(both_wines_array)

var_expl = pca.explained_variance_ratio_
var_cum_expl = np.cumsum(np.round(pca.explained_variance_ratio_,decimals=4)*100)
print(var_cum_expl)


# In[68]:

plt.plot(var_cum_expl)


# In[83]:

pca = PCA(n_components=2)
both_wines_array_red = pca.fit(both_wines_array).transform(both_wines_array)

print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy','turquoise']

for color, i, wine_type in zip(colors,[0,1,2]):
    plt.scatter(both_wines_array_red[both_wines_labels == i, 0],
                both_wines_array_red[both_wines_labels == i, 1],
                color=color,alpha=.8,label=wine_type)

plt.title('PCA of Wine dataset')

