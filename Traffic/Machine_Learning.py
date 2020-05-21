
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[4]:

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from matplotlib import pyplot as plt
from itertools import product
from scipy import optimize
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras import utils
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')


# In[5]:

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[6]:

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# In[7]:

AWS_ACCESS_KEY = "AKIA4JL5A5WRTIXMJ4AU"
AWS_SECRET_KEY = "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"


# In[8]:

get_ipython().run_cell_magic('time', '', 'import boto3\nimport io\nimport pandas as pd\n\ns3 = boto3.client(\'s3\', aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY)\n# Get binary objects\nbuffer2015= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2015/traffic\')["Body"].read())\nbuffer2016= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2016/traffic\')["Body"].read())\nbuffer2017= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2017/traffic\')["Body"].read())\nbuffer2018= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2018/traffic\')["Body"].read())\nbuffer2019= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2019/traffic\')["Body"].read())\nbuffer2020= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2020/traffic\')["Body"].read())')


# In[9]:

df_2015=pd.read_parquet(buffer2015)
df_2016=pd.read_parquet(buffer2016)
df_2017=pd.read_parquet(buffer2017)
df_2018=pd.read_parquet(buffer2018)
df_2019=pd.read_parquet(buffer2019)
df_2020=pd.read_parquet(buffer2020)


# ## <font color =blue>Filtering the Freeway

# In[10]:

df_2015=df_2015[df_2015['freeway'].isin([101,680,880,280])]
df_2016=df_2016[df_2016['freeway'].isin([101,680,880,280])]
df_2017=df_2017[df_2017['freeway'].isin([101,680,880,280])]
df_2018=df_2018[df_2018['freeway'].isin([101,680,880,280])]
df_2019=df_2019[df_2019['freeway'].isin([101,680,880,280])]
df_2020=df_2020[df_2020['freeway'].isin([101,680,880,280])]


# In[11]:

df_2019.freeway.value_counts()


# In[12]:

df_2015=df_2015[df_2015['direction']=='N']
df_2016=df_2016[df_2016['direction']=='N']
df_2017=df_2017[df_2017['direction']=='N']
df_2018=df_2018[df_2018['direction']=='N']
df_2019=df_2019[df_2019['direction']=='N']
df_2020=df_2020[df_2020['direction']=='N']


# In[13]:

df_2019.direction.value_counts()


# ## <font color=blue>Filtering only 101 Freeway

# In[14]:

df_101_2015=df_2015[df_2015['freeway']==101]
df_101_2016=df_2016[df_2016['freeway']==101]
df_101_2017=df_2017[df_2017['freeway']==101]
df_101_2018=df_2018[df_2018['freeway']==101]
df_101_2019=df_2019[df_2019['freeway']==101]
df_101_2020=df_2020[df_2020['freeway']==101]


# In[15]:

df_101_2019.head(20)


# ## <font color=blue> Filtering the 400001 Station id alone

# In[16]:

df_101_2015=df_101_2015[df_101_2015['station']==400001]
df_101_2016=df_101_2016[df_101_2016['station']==400001]
df_101_2017=df_101_2017[df_101_2017['station']==400001]
df_101_2018=df_101_2018[df_101_2018['station']==400001]
df_101_2019=df_101_2019[df_101_2019['station']==400001]
df_101_2020=df_101_2020[df_101_2020['station']==400001]


# In[17]:

df1_2015=df_101_2015.drop_duplicates(subset=['station','timestamp_'])
df1_2016=df_101_2016.drop_duplicates(subset=['station','timestamp_'])
df1_2017=df_101_2017.drop_duplicates(subset=['station','timestamp_'])
df1_2018=df_101_2018.drop_duplicates(subset=['station','timestamp_'])
df1_2019=df_101_2019.drop_duplicates(subset=['station','timestamp_'])
df1_2020=df_101_2020.drop_duplicates(subset=['station','timestamp_'])



# In[18]:

df1_2015= (df1_2015.set_index(['station','timestamp_']).sort_values(['station','timestamp_']))
df1_2016= (df1_2016.set_index(['station','timestamp_']).sort_values(['station','timestamp_']))
df1_2017= (df1_2017.set_index(['station','timestamp_']).sort_values(['station','timestamp_']))
df1_2018= (df1_2018.set_index(['station','timestamp_']).sort_values(['station','timestamp_']))
df1_2019= (df1_2019.set_index(['station','timestamp_']).sort_values(['station','timestamp_']))
df1_2020= (df1_2020.set_index(['station','timestamp_']).sort_values(['station','timestamp_']))




# In[19]:

df1_2015.reset_index(inplace=True)
df1_2016.reset_index(inplace=True)
df1_2017.reset_index(inplace=True)
df1_2018.reset_index(inplace=True)
df1_2019.reset_index(inplace=True)
df1_2020.reset_index(inplace=True)


# In[20]:

df1_2019.head()


# ## <font color=blue> Considering the day_of_week_num and hour_of_day as features

# In[21]:

cols = ['occupancy','speed','day_of_week_num','hour_of_day']
df_2015=df1_2015[cols]
df_2016=df1_2016[cols]
df_2017=df1_2017[cols]
df_2018=df1_2018[cols]
df_2019=df1_2019[cols]
df_2020=df1_2020[cols]


# In[22]:

df_2019.head()


# In[23]:

df_2015.head()


# In[24]:

data=pd.concat([df_2015,df_2016,df_2017,df_2018,df_2019])
data_predict=df_2020


# In[25]:

data_predict.shape


# ## <font color=blue> Linear Regression

# In[26]:

from sklearn.model_selection import TimeSeriesSplit

train=data.values
test=data_predict.values

# split into input and outputs
train_X, train_y = train[:, [0,2,3]], train[:,1]
#val_X, val_y = val[:, [0,2,3]], val[:,1]
test_X, test_y = test[:,[0,2,3]], test[:,1]

print(train_X.shape,train_y.shape)
print(test_X.shape,test_y.shape)


# In[27]:

# Import library for Linear Regression
from sklearn.linear_model import LinearRegression

# Create a Linear regressor
lm = LinearRegression()

# Train the model using the training sets 
lm.fit(train_X, train_y)


# In[28]:

# Predicting Test data with the model
y_test_pred = lm.predict(test_X)


# In[29]:

from sklearn import metrics
from sklearn.metrics import accuracy_score


# Model Evaluation
print('R^2:',metrics.r2_score(test_y, y_test_pred))
#print('Adjusted R^2:',1 - (1-metrics.r2_score(train_y, y_pred))*(len(train_y)-1)/(len(train_y)-train_X.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(test_y, y_test_pred))
print('MAPE:',mean_absolute_percentage_error(test_y, y_test_pred))
print('MSE:',metrics.mean_squared_error(test_y, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(test_y, y_test_pred)))


# In[30]:

from sklearn import metrics
from sklearn.metrics import accuracy_score



# In[31]:

y_actual=pd.DataFrame(test_y)
y_predicted=pd.DataFrame((y_test_pred))
df_2020_s=pd.concat([y_actual,y_predicted],axis=1)

col=['y_actual','y_predicted']
df_2020_s.columns=col
#df_2020.columns=['y_actual','y_predicted']
df_2020_s['y_predicted']=df_2020_s['y_predicted'].round(1)

df_2020_s.reset_index()
df_2020_s.head(50)


# In[32]:

linear_1year=pd.concat([df_2020,df_2020_s],axis=1)


# In[33]:

linear_1year.head(50)


# In[34]:

linear_1year.to_csv('linear_5year.csv')


# ## <font color=blue> Random Forest Regressor

# In[35]:

# Import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
params={'min_samples_split': 10, 'max_depth': 110, 'max_features': 2, 'n_estimators': 100, 'min_samples_leaf': 5, 'bootstrap': True}
# Create a Random Forest Regressor
reg = RandomForestRegressor(**params)

# Train the model using the training sets 
reg.fit(train_X, train_y)


# In[38]:

import pickle
# save the model to disk
filename = 'Random_model'
pickle.dump(reg, open(filename, 'wb'))


# In[39]:

# load the model from disk
random_model = pickle.load(open(filename, 'rb'))


# In[40]:


# Predicting Test data with the model
y_test_pred = random_model.predict(test_X)


# In[41]:

# Model Evaluation
print('R^2:',metrics.r2_score(test_y, y_test_pred))
#print('Adjusted R^2:',1 - (1-metrics.r2_score(train_y, y_pred))*(len(train_y)-1)/(len(train_y)-train_X.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(test_y, y_test_pred))
print('MAPE:',mean_absolute_percentage_error(test_y, y_test_pred))
print('MSE:',metrics.mean_squared_error(test_y, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(test_y, y_test_pred)))


# In[42]:

y_actual=pd.DataFrame(test_y)
y_predicted=pd.DataFrame((y_test_pred))
df_2020_s=pd.concat([y_actual,y_predicted],axis=1)

col=['y_actual','y_predicted']
df_2020_s.columns=col
#df_2020.columns=['y_actual','y_predicted']
df_2020_s['y_predicted']=df_2020_s['y_predicted'].round(1)

df_2020_s.reset_index()
df_2020_s.head(50)


# In[43]:

Randomforest_1year=pd.concat([df_2020,df_2020_s],axis=1)


# In[44]:

Randomforest_1year.head(50)


# In[45]:

Randomforest_1year.to_csv('Randomforest_5year.csv')


# In[46]:

# Commented after running on a local computer
# takes a longer time
'''
from sklearn.model_selection import GridSearchCV

# parameters for GridSearchCV
param_grid2 = {"n_estimators": [10, 18, 22],
              "max_depth": [3, 5],
              "min_samples_split": [15, 20],
              "min_samples_leaf": [5, 10, 20],
              "max_leaf_nodes": [20, 40],
              "min_weight_fraction_leaf": [0.1]}
grid_search = GridSearchCV(model2, param_grid=param_grid2)
grid_search.fit(train_X, np.log1p(train_y['visitors'].values))
'''
from operator import itemgetter

# Utility function to report best scores
def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
# Commented after running on a local computer
#report(grid_search.grid_scores_,4)


# ## <font color=blue> AdaBoost Regressor

# In[47]:

from sklearn.ensemble import AdaBoostRegressor
ADB = AdaBoostRegressor()
ADB.fit(train_X, train_y)
y_test_pred=ADB.predict(test_X)


# In[48]:

#Model Evaluation
print('R^2:',metrics.r2_score(test_y, y_test_pred))
#print('Adjusted R^2:',1 - (1-metrics.r2_score(train_y, y_pred))*(len(train_y)-1)/(len(train_y)-train_X.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(test_y, y_test_pred))
print('MAPE:',mean_absolute_percentage_error(test_y, y_test_pred))
print('MSE:',metrics.mean_squared_error(test_y, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(test_y, y_test_pred)))


# In[49]:

y_actual=pd.DataFrame(test_y)
y_predicted=pd.DataFrame((y_test_pred))
df_2020_s=pd.concat([y_actual,y_predicted],axis=1)

col=['y_actual','y_predicted']
df_2020_s.columns=col
#df_2020.columns=['y_actual','y_predicted']
df_2020_s['y_predicted']=df_2020_s['y_predicted'].round(1)

df_2020_s.reset_index()
df_2020_s.head(50)


# In[50]:

AdaBoost_1year=pd.concat([df_2020,df_2020_s],axis=1)


# In[51]:

AdaBoost_1year.head(20)


# In[52]:

AdaBoost_1year.to_csv("AdaBoost_5year.csv")


# ## <font color=blue> XGBoost Regressor

# In[53]:

from sklearn.ensemble import GradientBoostingRegressor
GB= GradientBoostingRegressor()
GB.fit(train_X, train_y)


# In[56]:

# save the model to disk
filename = 'GB_model'
pickle.dump(GB, open(filename, 'wb'))


# In[57]:

# load the model from disk
GB_model = pickle.load(open(filename, 'rb'))


# In[58]:

y_test_pred=GB_model.predict(test_X)


# In[59]:

#Model Evaluation
print('R^2:',metrics.r2_score(test_y, y_test_pred))
#print('Adjusted R^2:',1 - (1-metrics.r2_score(train_y, y_pred))*(len(train_y)-1)/(len(train_y)-train_X.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(test_y, y_test_pred))
print('MAPE:',mean_absolute_percentage_error(test_y, y_test_pred))
print('MSE:',metrics.mean_squared_error(test_y, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(test_y, y_test_pred)))


# In[60]:

y_actual=pd.DataFrame(test_y)
y_predicted=pd.DataFrame((y_test_pred))
df_2020_s=pd.concat([y_actual,y_predicted],axis=1)

col=['y_actual','y_predicted']
df_2020_s.columns=col
#df_2020.columns=['y_actual','y_predicted']
df_2020_s['y_predicted']=df_2020_s['y_predicted'].round(1)

df_2020_s.reset_index()
df_2020_s.head(50)


# In[61]:

XGBoost_1year=pd.concat([df_2020,df_2020_s],axis=1)


# In[62]:

XGBoost_1year.head(20)


# In[63]:

XGBoost_1year.to_csv("XGBoost_5year.csv")


# ## <font color=blue> Support vector Regressor

# In[64]:

# Import SVM Regressor
from sklearn import svm

# Create a SVM Regressor
svm = svm.SVR()
svm.fit(train_X, train_y)


# In[65]:

# Predicting Test data with the model
y_test_pred = svm.predict(test_X)



# In[66]:

#Model Evaluation
print('R^2:',metrics.r2_score(test_y, y_test_pred))
#print('Adjusted R^2:',1 - (1-metrics.r2_score(train_y, y_pred))*(len(train_y)-1)/(len(train_y)-train_X.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(test_y, y_test_pred))
print('MAPE:',mean_absolute_percentage_error(test_y, y_test_pred))
print('MSE:',metrics.mean_squared_error(test_y, y_test_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(test_y, y_test_pred)))


# In[67]:

y_actual=pd.DataFrame(test_y)
y_predicted=pd.DataFrame((y_test_pred))
df_2020_s=pd.concat([y_actual,y_predicted],axis=1)

col=['y_actual','y_predicted']
df_2020_s.columns=col
#df_2020.columns=['y_actual','y_predicted']
df_2020_s['y_predicted']=df_2020_s['y_predicted'].round(1)

df_2020_s.reset_index()
df_2020_s.head(50)


# In[68]:

SVR_1year=pd.concat([df_2020,df_2020_s],axis=1)


# In[69]:

SVR_1year.head(20)


# In[70]:

SVR_1year.to_csv("SVR_5year.csv")


# ## <font color=blue> Advanced Ensemble Methods - Stacking Technique

# In[71]:

# compare machine learning models for regression
from numpy import mean
from numpy import std
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
#from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import StackingRegressor

from sklearn.svm import SVR
from matplotlib import pyplot


# In[ ]:




# In[72]:

# evaluate a given model using cross-validation
def evaluate_model(model):
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
	return scores


# In[ ]:




# In[73]:


# get a stacking ensemble of models
def get_stacking():
# define the base models
    level0 = list()
    level0.append(('knn', KNeighborsRegressor()))
    level0.append(('cart', DecisionTreeRegressor()))
    level0.append(('svm', SVR()))
    level0.append(('rf',RandomForestRegressor(**params)))
# define meta learner model
    level1 = LinearRegression()
	# define the stacking ensemble
    model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
    return model


# In[74]:

from sklearn.ensemble import StackingRegressor


# In[75]:

# get a list of models to evaluate
def get_models():
    models = dict()
    models['knn'] = KNeighborsRegressor()
    models['cart'] = DecisionTreeRegressor()
    models['svm'] = SVR()
    models['rf']=RandomForestRegressor()
    models['stacking'] = get_stacking()
    return models


# In[ ]:




# In[76]:

# make a prediction with a stacking ensemble
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
# define dataset
X, y = train_X,train_y
# define the base models
level0 = list()
level0.append(('knn', KNeighborsRegressor()))
level0.append(('cart', DecisionTreeRegressor()))
level0.append(('svm', SVR()))
level0.append(("rf",RandomForestRegressor()))
# define meta learner model
level1 = LinearRegression()
# define the stacking ensemble
model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
# fit the model on all available data
model.fit(X, y)
# make a prediction for one example


# In[77]:

import pickle
# save the model to disk
filename = 'ensemble_model'
pickle.dump(model, open(filename, 'wb'))
 


# In[78]:

# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
yhat = loaded_model.predict(test_X)


# In[79]:

yhat = model.predict(test_X)


# In[80]:

#Model Evaluation
print('R^2:',metrics.r2_score(test_y, yhat))
#print('Adjusted R^2:',1 - (1-metrics.r2_score(train_y, y_pred))*(len(train_y)-1)/(len(train_y)-train_X.shape[1]-1))
print('MAE:',metrics.mean_absolute_error(test_y, yhat))
print('MAPE:',mean_absolute_percentage_error(test_y, yhat))
print('MSE:',metrics.mean_squared_error(test_y, yhat))
print('RMSE:',np.sqrt(metrics.mean_squared_error(test_y, yhat)))


# In[81]:

y_actual=pd.DataFrame(test_y)
y_predicted=pd.DataFrame((yhat))
df_2020_s=pd.concat([y_actual,y_predicted],axis=1)

col=['y_actual','y_predicted']
df_2020_s.columns=col
#df_2020.columns=['y_actual','y_predicted']
df_2020_s['y_predicted']=df_2020_s['y_predicted'].round(1)

df_2020_s.reset_index()
df_2020_s.head(50)


# In[82]:

ensemble_1year=pd.concat([df_2020,df_2020_s],axis=1)


# In[83]:

ensemble_1year.head()


# In[84]:

ensemble_1year.to_csv("ensemble_5year.csv")


# ## <font color=blue> References
# 
# 1.https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/     
# 2.https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/
# 

# In[ ]:



