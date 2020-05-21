
# coding: utf-8

# In[43]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[44]:

df1=pd.read_csv("Randomforest_5year.csv")
df2=pd.read_csv("XGBoost_5year.csv")
df3=pd.read_csv("ensemble_5year.csv")
df4=pd.read_csv("LSTM.csv")
df5=pd.read_csv("GRU.csv")


# In[45]:

df1 =df1.drop(columns=['Unnamed: 0'])
df2 =df2.drop(columns=['Unnamed: 0'])
df3 =df3.drop(columns=['Unnamed: 0'])
df4 =df4.drop(columns=['Unnamed: 0'])
df5 =df5.drop(columns=['Unnamed: 0'])


# In[46]:

df1.head()


# In[47]:

df1.shape


# In[48]:

df5.shape


# In[49]:

df1.rename(columns={'y_predicted':"y_RandomForest"},inplace=True)
df2.rename(columns={'y_predicted':"y_XGBoost"},inplace=True)
df3.rename(columns={'y_predicted':"y_Ensemble"},inplace=True)
df4.rename(columns={'y_predicted':"y_LSTM"},inplace=True)
df5.rename(columns={'y_predicted':"y_GRU"},inplace=True)


# In[50]:

df1.head()


# In[ ]:




# In[51]:

df1=df1[3:]
df1.reset_index(inplace=True,drop=True)


# In[52]:

df1.head()


# In[53]:

df1.tail()


# In[54]:

df4.tail()


# In[55]:

df1.shape


# In[42]:




# ## <font color=blue> Since LSTM and GRU miss three points

# In[56]:

df2=df2[3:]
df2.reset_index(inplace=True,drop=True)

df3=df3[3:]
df3.reset_index(inplace=True,drop=True)




# In[57]:

df= pd.concat([df1,df2['y_XGBoost'],df3['y_Ensemble'],df4['y_LSTM'],df5['y_GRU']],axis=1, join='inner')
df.head()



# In[58]:

df.tail()


# In[59]:

df.head(50)


# In[60]:

AWS_ACCESS_KEY = "AKIA4JL5A5WRTIXMJ4AU"
AWS_SECRET_KEY = "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"


# In[61]:

get_ipython().run_cell_magic('time', '', 'import boto3\nimport io\nimport pandas as pd\n\ns3 = boto3.client(\'s3\', aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY)\nbuffer2020= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2020/traffic\')["Body"].read())')


# In[62]:

df_2020=pd.read_parquet(buffer2020)
df_2020=df_2020[df_2020['freeway'].isin([101,680,880,280])]
df_2020=df_2020[df_2020['direction']=='N']
df_101_2020=df_2020[df_2020['freeway']==101]
df_week_2020=df_101_2020[df_101_2020['station']==400001]


# In[63]:

df1_2020=df_week_2020.drop_duplicates(subset=['station','timestamp_'])

df1_2020.head()


# In[64]:

df1_2020.reset_index(inplace = True) 


# In[65]:

df1_2020.head()


# In[66]:

cols = ['station','timestamp_','occupancy','speed','latitude','longitude','day_of_week',"day_of_week_num","hour_of_day"]
df1_2020= (df1_2020[cols].set_index(['station','timestamp_']).sort_values(['station','timestamp_']))





# In[67]:

df1_2020.reset_index(inplace = True) 


# In[68]:

df1_2020.head()


# In[69]:

df1_2020.shape


# In[70]:

df1_2020=df1_2020[3:]
df1_2020.reset_index(inplace=True,drop=True)


# In[72]:

final=pd.concat([df1_2020, df[['y_actual','y_RandomForest','y_XGBoost','y_Ensemble','y_LSTM','y_GRU']]],axis=1, join='inner')


# In[73]:

final.head()


# In[74]:

final.to_csv("graph_400001.CSV")


# In[ ]:



