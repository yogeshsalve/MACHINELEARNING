#!/usr/bin/env python
# coding: utf-8

# <h2 style="color:green" align="center"> Machine Learning With Python: Linear Regression Multiple Variables</h2>

# ***In this machine learning tutorial with python, we will write python code to predict home prices using multivariate linear regression in python (using sklearn linear_model). Home prices are dependent on 3 independent variables: area, bedrooms and age. Pandas dataframe is used to fill missing values first and then use that dataset to train a multivariate regression model.***

# In[72]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# **READING .CSV FILE AND SHOWING FIRST 5 ROWS**

# In[74]:


df=pd.read_csv("homeprices.csv")
df.head()


# **Preprocessing - for missing values if any**

# In[77]:


import math
median_bedrooms=math.floor(df.bedrooms.median())
median_bedrooms


# In[78]:


df.bedrooms=df.bedrooms.fillna(median_bedrooms)
df


# **Create Linear Regression Model**

# In[79]:


reg=linear_model.LinearRegression()
reg.fit(df[['area','bedrooms','age']],df.price)


# **For Predictions**

# **Find price of home with 3000 sqr ft area, 3 bedrooms, 40 year old**

# In[80]:


reg.predict([[3000,3,40]])


# **PLOTTING DATA**

# In[83]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Area, Bedrooms Age')
plt.ylabel('Price')
plt.scatter(df.area, df.price, color='red', marker='+')


# **CHECK COEFFICIENT AND INTERCEPTION ELEMENTS FOR REGRESSION EQUATION**

# In[84]:


reg.coef_


# In[85]:


reg.intercept_


# **VISUAL REPRESENTATION OF MY LINEAR REGRESSION MODEL**

# In[87]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('Area, Bedrooms Age')
plt.ylabel('Price')
plt.scatter(df.area, df.price, color='red', marker='+')
plt.plot(df.area, reg.predict(df[['area','bedrooms','age']]), color='blue')

