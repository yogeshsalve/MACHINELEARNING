#!/usr/bin/env python
# coding: utf-8

# ***A LINEAR REGRESSION MODEL TO PREDICT VALUE BASED ON PROVIDED INPUT***

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# **READING .CSV FILE AND SHOWING FIRST 5 ROWS**

# In[6]:


df=pd.read_csv("model1.csv")
df.head()


# **PLOTTING DATA**

# In[28]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(df.x, df.y, color='red', marker='+')


# **TRAINING MODEL**

# In[29]:


reg=linear_model.LinearRegression()
reg.fit(df[['x']],df.y)


# **PREDICTION MODEL**

# In[30]:


reg.predict([[26]])


# **CHECK COEFFICIENT AND INTERCEPTION ELEMENTS FOR REGRESSION EQUATION**

# In[31]:


reg.coef_


# In[32]:


reg.intercept_

