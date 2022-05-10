#!/usr/bin/env python
# coding: utf-8

# ## DOMAIN:
# 
# 

# ### Sports Management

# DATA DESCRIPTION:
# 
# The data is collected belongs to batsman from IPL series conducted so far. Attribute Information:
# 
#     Runs: Runs score by the batsman
#     Ave: Average runs scored by the batsman per match
#     SR: strike rate of the batsman
#     Fours: number of boundary/four scored
#     Six: number of boundary/six scored
#     HF: number of half centuries scored so far
# 

# PROJECT OBJECTIVE:

# Goal is to 
# build a data driven batsman ranking model for the sports management company to make business decisions.

# In[1]:


#loading the required packages
import numpy as np   
import pandas as pd    
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from scipy.stats import zscore
from sklearn.metrics import silhouette_samples, silhouette_score
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(color_codes=True)
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('df_bat.csv')
df.dropna(inplace=True)


# In[8]:


df.reset_index(drop=True, inplace=True)


# In[9]:


df.columns


# In[11]:


df


# In[13]:


##EDA


# In[10]:


#checking data types
df.dtypes


# In[12]:


#checking for the dimension of the data
rows, column = df.shape
print('The dataset contains', rows, 'rows and', column, 'columns.')


# In[15]:


#5-point summary
df.describe().T


# In[16]:


print('Duplicated rows: ', df[df.duplicated()].shape[0])


# In[17]:


plt.figure(figsize=(15,8))
b1=df.groupby('Name')['Runs'].sum().sort_values(ascending = False ).head(10)
b1= b1.reset_index() 
b1.columns = ['Name' ,'Runs']
sns.barplot(data= b1 , x= 'Name' , y ='Runs');
plt.title("Top 10 Players by Runs");


# In[18]:


plt.figure(figsize=(15,8))
b1=df.groupby('Name')['Ave'].sum().sort_values(ascending = False ).head(10)
b1= b1.reset_index() 
b1.columns = ['Name' ,'Ave']
sns.barplot(data= b1 , x= 'Name' , y ='Ave')
plt.title("Top 10 players by Average");


# In[19]:


plt.figure(figsize=(15,8))
b1=df.groupby('Name')['SR'].sum().sort_values(ascending = False ).head(10)
b1= b1.reset_index() 
b1.columns = ['Name' ,'SR']
sns.barplot(data= b1 , x= 'Name' , y ='SR')
plt.title("Top 10 players by Strike Rate");


# In[20]:


plt.figure(figsize=(15,8))
b1=df.groupby('Name')['Fours'].sum().sort_values(ascending = False ).head(10)
b1= b1.reset_index() 
b1.columns = ['Name' ,'Fours']
sns.barplot(data= b1 , x= 'Name' , y ='Fours')
plt.title("Top 10 players by Fours");


# In[21]:


plt.figure(figsize=(15,8))
b1=df.groupby('Name')['Sixes'].sum().sort_values(ascending = False ).head(10)
b1= b1.reset_index() 
b1.columns = ['Name' ,'Sixes']
sns.barplot(data= b1 , x= 'Name' , y ='Sixes')
plt.title("Top 10 players by Sixes");


# In[22]:


plt.figure(figsize=(15,8))
b1=df.groupby('Name')['HF'].sum().sort_values(ascending = False ).head(10)
b1= b1.reset_index() 
b1.columns = ['Name' ,'HF']
sns.barplot(data= b1 , x= 'Name' , y ='HF')
plt.title("Top 10 players by Half Centuries");


# In[23]:


df.hist(bins = 20, figsize = (15, 10), color = 'blue')
plt.show()


# In[25]:


plt.figure(figsize=(15, 12))
col = 1
for i in df.drop(columns='Name').columns:
    plt.subplot(2, 3, col)
    sns.distplot(df[i], color = 'b')
    col += 1 


# In[27]:


#Strike rate, fours, sixes and half centuries have a skewed distribution


# In[28]:


plt.figure(figsize=(15, 10))
col = 1
for i in df.drop(columns='Name').columns:
    plt.subplot(2, 3, col)
    sns.boxplot(df[i],color='blue')
    col += 1


# In[30]:




#There appears to be outliers, will not be treating them as its highly likely that these are 
#genuine observation


# In[32]:


#checking for correlation
plt.figure(figsize=(10,8))
corr=df.drop(columns='Name').corr()
sns.heatmap(corr,annot=True);


# In[33]:




#All the variable except fours with strike rate, strike rate with half centuries,
#strike rate with runs, have high correlation


# In[34]:


cc = df.iloc[:,1:7] 
cc1 = cc.apply(zscore)
cc1.head()


# In[35]:


#checking for the within sum of squares
wss =[] 
for i in range(1,6):
    KM = KMeans(n_clusters=i)
    KM.fit(cc1)
    wss.append(KM.inertia_)
wss


# In[36]:


#plotting to check for optimal clustres 
plt.plot(range(1,6), wss);
plt.title('Elbow Method');
plt.xlabel("Number of Clusters")
plt.ylabel("WSS");


# In[37]:


#using 2 centroids
k_means = KMeans(n_clusters = 2)
k_means.fit(cc1)
labels = k_means.labels_


# In[38]:


# Calculating silhouette_score
silhouette_score(cc1,labels)


# In[39]:


#plotting silhouette score for different centroids
kmeans_kwargs = {
   "init": "random",
   "n_init": 10,
   "max_iter": 300,
   "random_state": 42,
}


silhouette_coefficients = []

 # Notice you start at 2 clusters for silhouette coefficient
for k in range(2, 6):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(cc1)
    score = silhouette_score(cc1,kmeans.labels_)
    silhouette_coefficients.append(score)
    


# In[40]:


plt.plot(range(2,6), silhouette_coefficients)
plt.xticks(range(2, 6))
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Coefficient")
plt.show()


# In[43]:


#attaching the labels to the original data
df['cluster']=labels
df.head()


# In[45]:


#veiwing the distribution of the clusters
df.cluster.value_counts().sort_index()




# In[47]:


#aggregating the clusters with the numeric variables with their mean
aggdata=df.iloc[:,0:9].groupby('cluster').mean()
aggdata['Freq']=df.cluster.value_counts().sort_index()
aggdata


# In[51]:


#based on the above table renaming/ranking the playes in Grade A and Grade B
df['cluster'] = df['cluster'].replace({0: 'Grade A', 1: 'Grade B'})


# In[52]:


#list of Grade A players
Grade_A = df[df['cluster'] == 'Grade A']
Grade_A.head(10)


# In[53]:


#list of Grade B players
Grade_B = df[df['cluster'] == 'Grade B']
Grade_B.head(10)


# In[ ]:




