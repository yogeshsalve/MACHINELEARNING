#!/usr/bin/env python
# coding: utf-8

# ***A LINEAR REGRESSION MODEL TO PREDICT CANADA PER CAPITA INCOME***

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# **READING .CSV FILE AND SHOWING FIRST 5 ROWS**

# In[7]:


df=pd.read_csv("canada_per_capita_income.csv")
df.head()


# **PLOTTING DATA**

# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year')
plt.ylabel('per capita income(US$)')
plt.scatter(df.year, df.percapitaincome, color='red', marker='+')


# **TRAINING MODEL**

# In[9]:


reg=linear_model.LinearRegression()
reg.fit(df[['year']],df.percapitaincome)


# **PREDICTION MODEL**

# In[10]:


reg.predict([[2020]])


# **VISUAL REPRESENTATION OF MY LINEAR REGRESSION MODEL FOR CANDA PER CAPITA INCOME**

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('year')
plt.ylabel('per capita income(US$)')
plt.scatter(df.year, df.percapitaincome, color='red', marker='+')
plt.plot(df.year, reg.predict(df[['year']]), color='blue')

