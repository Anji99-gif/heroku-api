import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_excel('data_science_sample_data_v2.xlsx')


# In[49]:


df.head()


# In[5]:


df.dtypes


# In[7]:


df.shape


# In[6]:


df.isnull().sum()


# In[8]:


df.drop('Application_ID',axis=1,inplace=True)


# In[10]:


df1=df.copy()


# In[11]:


df1


# In[213]:


df1['Other skills'].mode()


# In[214]:


df1['Other skills'].fillna(df1['Other skills'].mode()[0],inplace=True)


# In[25]:


df1['Degree'].fillna(df1.Degree.value_counts().idxmax(),inplace=True)


# In[35]:


df1['Stream'].fillna(df1['Stream'].value_counts().idxmax(),inplace=True)


# In[46]:


df1['p_pg']=df1['Performance_PG'].apply(lambda x:x.split('/')[0])


# In[41]:


df1['Performance_PG'].fillna(df1['Performance_PG'].value_counts().idxmax(),inplace=True)


# In[51]:


df1['Performance_UG'].fillna(df1['Performance_UG'].value_counts().idxmax(),inplace=True)


# In[53]:


df1['Performance_12'].fillna(df1['Performance_12'].value_counts().idxmax(),inplace=True)
df1['Performance_10'].fillna(df1['Performance_10'].value_counts().idxmax(),inplace=True)


# In[58]:


df1.head()


# In[56]:


df1.dtypes


# In[63]:


df1['outof_pg']=df1['Performance_PG'].apply(lambda x:x.split('/')[1])


# In[66]:


df1['p_ug']=df1['Performance_UG'].apply(lambda x:x.split('/')[0])
df1['outof_ug']=df1['Performance_UG'].apply(lambda x:x.split('/')[1])


# In[67]:


df1['p_12']=df1['Performance_12'].apply(lambda x:x.split('/')[0])
df1['outof_12']=df1['Performance_12'].apply(lambda x:x.split('/')[1])
df1['p_10']=df1['Performance_10'].apply(lambda x:x.split('/')[0])
df1['outof_10']=df1['Performance_10'].apply(lambda x:x.split('/')[1])


# In[68]:


df1


# In[70]:


df1.drop(['Performance_UG','Performance_PG','Performance_12','Performance_10'],axis=1,inplace=True)


# In[86]:


df1['Current City'].value_counts().plot(kind='bar',figsize=(14,10),grid=True)


# In[88]:


df1.drop(['Current City'],axis=1,inplace=True)


# In[96]:


df1.drop('Institute',axis=1,inplace=True)


# In[99]:


df1


# In[100]:


df2=df1.copy()


# In[98]:


otherskills = df1['Other skills']
otherskills


# In[102]:


df2.drop('Other skills',axis=1,inplace=True)


# In[103]:


df2


# In[104]:


from sklearn.preprocessing import LabelEncoder


# In[105]:


lbe=LabelEncoder()


# In[106]:


df2['Degree']=lbe.fit_transform(df2['Degree'])
df2['Stream']=lbe.fit_transform(df2['Stream'])


# In[108]:


df3=df2.copy()


# In[109]:


df3.head()


# In[121]:


from sklearn.preprocessing import StandardScaler


# In[122]:


std=StandardScaler()


# In[129]:


std_df=pd.DataFrame(std.fit_transform(df3),columns=df3.columns.tolist())
std_df


# In[91]:


from sklearn.cluster import KMeans


# In[378]:


km=KMeans(n_clusters=6,n_init=10,random_state=4)


# In[400]:


km.fit(std_df)


# In[402]:


centers=km.cluster_centers_
centers


# In[403]:


labels=km.predict(std_df)


# In[393]:


from sklearn.metrics import silhouette_score


# In[404]:


silhouette_score(df3,labels)


# In[409]:


pickle.dump(km,open("model.pkl","wb"))


# In[ ]:





# In[ ]:




