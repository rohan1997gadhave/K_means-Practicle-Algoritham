#!/usr/bin/env python
# coding: utf-8

#  ## Want to see customer  purchasing behavior

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# In[2]:


data = pd.read_csv("Mall_customers.csv")


# In[3]:


data.head()


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


# finding no of row and col
data.shape


# In[8]:


# copy real data set 
df = data.copy()


# In[9]:


df.shape


# In[10]:


# choosing Anual income and score column 
x = df.iloc[:,[3,4]].values


# ## Choosing no of Cluster 

# #### We will use  Wcss( within cluster sum of squares) call elbow method
# #### Try to find how much cluster will be the best for that data set. using elbow method 

# In[16]:


## finding teh cluster 
wcss = []
for i in range (1,11):
    kmeans = KMeans(n_clusters=i,init= 'k-means++',random_state= 42)
    kmeans.fit(x)
    
    wcss.append(kmeans.inertia_) ## will give wscc value for each cluster


# In[17]:


## finding elbow graph to see which cluster has minimum value 
sns.set()
plt.plot(range(1,11),wcss)
plt.title("The Elbow Graph")
plt.xlabel("No of Cluster")
plt.ylabel("Wcss")
plt.show()


# * Optimum no of Cluster will be 5 based on above fig

# ### Training k_means cluster Model 

# In[24]:


kmeans = KMeans(n_clusters =5,init="k-means++",random_state=0)


# #### Return the label based on there cluster 

# In[25]:


y = kmeans.fit_predict(x)
y


# In[28]:


## plot cluster along with their value 
plt.figure(figsize = (8,9))
plt.scatter(x[y==0,0],x[y==0,1],s= 50,c="green",label ="Cluster 1")
plt.scatter(x[y==1,0],x[y==1,1],s= 50,c="red",label ="Cluster 2")
plt.scatter(x[y==2,0],x[y==2,1],s= 50,c="yellow",label ="Cluster 3")
plt.scatter(x[y==3,0],x[y==3,1],s= 50,c="violet",label ="Cluster 4")
plt.scatter(x[y==4,0],x[y==4,1],s= 50,c="blue",label ="Cluster 5")


# plot centroid 
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], s=100,c='cyan',label = "Centroid")
plt.title("Custer Group")
plt.xlabel("Anual Income")
plt.ylabel("Spending Score")
plt.show()


# In[ ]:





# In[ ]:




