#!/usr/bin/env python
# coding: utf-8

# ## Assignment 3 - Abalone Age Prediction
# ## Importing Libraries

# In[3]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# ## 2.Load the dataset into the tool

# In[4]:


data = pd.read_csv('abalone.csv')
# getting the shape
data.shape


# In[5]:


# looking at the head of the data

data.head()


# ## 3.Perform Below Visualizatons
# 
# ## 3.1 Univariate Analysis

# In[6]:


data.hist(figsize=(20,10), grid=False, layout=(2, 4), bins = 30)


# In[7]:


numerical_features = data.select_dtypes(include=[np.number]).columns
categorical_features = data.select_dtypes(include=[np.object]).columns
numerical_features
categorical_features
sns.pairplot(data[numerical_features])


# ## 3.3 Multi-Variate Analysis

# In[8]:


plt.figure(figsize=(10, 10))
corr = data.corr()
_ = sns.heatmap(corr, annot=True)


# ## 4.Perform descriptive statistics on the dataset.

# In[10]:


data.describe()


# ## 5.Check for Missing values and deal with them.

# In[11]:


data.isnull().sum()


# ## 6.Find the outliers and replace them outliers

# In[12]:


data['Rings']=np.where(data['Rings']>10,np.median,data['Rings'])
data['Rings']


# ## 7.Check for Categorical columns and perform encoding.

# In[13]:


data.columns


# ## 8.Split the data into dependent and independent variables.
# 
# ## 8.1 Split the data in to Independent variables.

# In[14]:


X=data.iloc[:,:-2].values
print(X)


# ## 8.2 Split the data in to Dependent variables.

# In[17]:


Y=data.iloc[:,-1].values
print(Y)


# ## 9.Scale the independent variables

# In[18]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
data[["Length"]]=scaler.fit_transform(data[["Length"]])
print(data)


# ## 10.Split the data into training and testing

# In[19]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# getting the shapes
print("Shape of x_train :", X_train.shape)
print("Shape of x_test :", X_test.shape)
print("Shape of y_train :", Y_train.shape)
print("Shape of y_test :", Y_test.shape)


# ## 11.Build the model

# In[20]:


test_size=0.33
seed=7
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=test_size,random_state=seed)


# ## 12.Train the Model

# In[21]:


print(X_test)


# In[22]:


print(Y_train)


# ## 13.Test the Model

# In[23]:


print(X_test)


# In[24]:


print(Y_test)


# ## 14.Measure the performance using metrics

# In[25]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
X_train=[5,-1,2,10]
Y_test=[3.5,-0.9,2,9.9]
print('RSquared=',r2_score(X_train,Y_test))
print('MAE=',mean_absolute_error(X_train,Y_test))
print('MSE=',mean_squared_error(X_train,Y_test))


# In[ ]:




