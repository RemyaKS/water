#!/usr/bin/env python
# coding: utf-8

# In[3]:


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
data = pd.read_csv("water_potability.csv")
data.head()


# In[4]:


data = data.dropna()
data.isnull().sum()


# In[5]:


plt.figure(figsize=(15, 10))
sns.countplot(x='Potability', data=data)
plt.title('Distribution of Potability')
plt.show()


# In[6]:


import plotly.express as px
figure = px.histogram(data, x = "ph", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: PH")
figure.show()


# In[7]:


figure = px.histogram(data, x = "Hardness", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Hardness")
figure.show()


# In[8]:


figure = px.histogram(data, x = "Solids", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Solids")
figure.show()


# In[9]:


figure = px.histogram(data, x = "Chloramines", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Chloramines")
figure.show()


# In[10]:


figure = px.histogram(data, x = "Sulfate", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Sulfate")
figure.show()


# In[11]:


figure = px.histogram(data, x = "Conductivity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Conductivity")
figure.show()


# In[12]:


figure = px.histogram(data, x = "Organic_carbon", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Organic Carbon")
figure.show()


# In[13]:


figure = px.histogram(data, x = "Turbidity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Turbidity")
figure.show()


# In[ ]:





# In[44]:


pip install pycaret


# In[45]:


correlation = data.corr()
correlation["ph"].sort_values(ascending=False)


# In[53]:


from pycaret.classification import*  
clf = setup(data, target = "Potability",Prsession_id = 786)
compare_models()


# In[49]:


model = create_model("rf")
predict = predict_model(model, data=data)
predict.head()


# In[ ]:




