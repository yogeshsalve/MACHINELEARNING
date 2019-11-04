#!/usr/bin/env python
# coding: utf-8

# ***A LINEAR REGRESSION MODEL TO PREDICT VALUE BASED ON PROVIDED INPUT***

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# **READING .CSV FILE AND SHOWING FIRST 5 ROWS**

# In[3]:


df=pd.read_csv("model1.csv")
df.head()


# **PLOTTING DATA**

# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(df.x, df.y, color='red', marker='+')


# **TRAINING MODEL**

# In[5]:


reg=linear_model.LinearRegression()
reg.fit(df[['x']],df.y)


# **PREDICTION MODEL**

# In[6]:


reg.predict([[31]])


# **CHECK COEFFICIENT AND INTERCEPTION ELEMENTS FOR REGRESSION EQUATION**

# In[31]:


reg.coef_


# In[32]:


reg.intercept_


# **Now Generate a new .csv file with predicted value of y where only x are provided** 

# In[7]:


d= pd.read_csv("modelx.csv")
d.head(3)


# In[8]:


p=reg.predict(d)


# In[9]:


d['y']=p
d.to_csv("modelx.csv", index=False)


# **VISUAL REPRESENTATION OF MY LINEAR REGRESSION MODEL**

# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(df.x, df.y, color='red', marker='+')
plt.plot(df.x, reg.predict(df[['x']]), color='blue')

