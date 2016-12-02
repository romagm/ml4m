
# coding: utf-8

# # Exercise 8 Part a

# #### Imports

# In[1]:

get_ipython().magic(u'matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# #### Import Dataset

# In[2]:

red_wine = pd.read_csv('winequality-red.csv',sep=";")
white_wine = pd.read_csv('winequality-white.csv',sep=";")


# #### Overview of the datasets via .head() and .describe()

# In[3]:

red_wine.head()


# In[4]:

white_wine.head()


# In[5]:

red_wine.describe()


# In[6]:

white_wine.describe()


# #### Plotting the means (and the quartiles) of red and white wine in respect to the variables

# In[34]:

from pandas.tools.plotting import parallel_coordinates
fig.clf()

red_wine_desc = red_wine.describe()
white_wine_desc = white_wine.describe()
red_wine_desc_min = red_wine_desc.drop(['count','std','min','max'])
white_wine_desc_min = white_wine_desc.drop(['count','std','min','max'])

red_wine_desc_min['type']='red wine'
white_wine_desc_min['type']='white wine'
both_wines_desc_min = pd.concat([red_wine_desc_min,white_wine_desc_min],axis=0)
print(both_wines_desc_min)
fig = plt.figure(1,figsize=(20,10))
parallel_coordinates(both_wines_desc_min,'type')


# #### There's definitely some separation possible with the variables 'free sulfur dioxide' and 'total sulfur dioxide'
# But let's drop these two variables to see if the other variables have similar discrepancies between the types.

# In[35]:

from pandas.tools.plotting import parallel_coordinates
fig.clf()

both_wines_desc_min_nar = both_wines_desc_min.drop(['free sulfur dioxide','total sulfur dioxide'],axis=1)
fig = plt.figure(2,figsize=(20,10))
parallel_coordinates(both_wines_desc_min_nar,'type')


# #### Most other variables are not really different enough, except maybe for 'residual sugar', but there you can see that the 25% percentile of the white wines are below the equivalent of the red wines, so that a higher value indicates white wine but a lower value doesn't mean a lot.
