
# coding: utf-8

# In[23]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[24]:

AWS_ACCESS_KEY = "AKIA4JL5A5WRTIXMJ4AU"
AWS_SECRET_KEY = "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"


# In[25]:

get_ipython().run_cell_magic('time', '', 'import boto3\nimport io\nimport pandas as pd\n\ns3 = boto3.client(\'s3\', aws_access_key_id=AWS_ACCESS_KEY,aws_secret_access_key=AWS_SECRET_KEY)\n# Get binary objects\nbuffer2015= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2015/traffic\')["Body"].read())\nbuffer2016= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2016/traffic\')["Body"].read())\nbuffer2017= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2017/traffic\')["Body"].read())\nbuffer2018= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2018/traffic\')["Body"].read())\nbuffer2019= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2019/traffic\')["Body"].read())\nbuffer2020= io.BytesIO(s3.get_object(Bucket=\'pemstraffic\', Key=\'2020/traffic\')["Body"].read())')


# In[26]:

df_2015=pd.read_parquet(buffer2015)
df_2016=pd.read_parquet(buffer2016)
df_2017=pd.read_parquet(buffer2017)
df_2018=pd.read_parquet(buffer2018)
df_2019=pd.read_parquet(buffer2019)
df_2020=pd.read_parquet(buffer2020)


# In[27]:

df_2019.shape


# ## <font color =blue>Filtering the Freeway

# In[28]:

df_2015=df_2015[df_2015['freeway'].isin([101,680,880,280])]
df_2016=df_2016[df_2016['freeway'].isin([101,680,880,280])]
df_2017=df_2017[df_2017['freeway'].isin([101,680,880,280])]
df_2018=df_2018[df_2018['freeway'].isin([101,680,880,280])]
df_2019=df_2019[df_2019['freeway'].isin([101,680,880,280])]
df_2020=df_2020[df_2020['freeway'].isin([101,680,880,280])]


# In[29]:

df_2019.freeway.value_counts()


# In[30]:

df_2015=df_2015[df_2015['direction']=='N']
df_2016=df_2016[df_2016['direction']=='N']
df_2017=df_2017[df_2017['direction']=='N']
df_2018=df_2018[df_2018['direction']=='N']
df_2019=df_2019[df_2019['direction']=='N']
df_2020=df_2020[df_2020['direction']=='N']


# In[31]:

df_2019.direction.value_counts()


# ## <font color=blue>Filtering only 101 Freeway

# In[32]:

df_101_2015=df_2015[df_2015['freeway']==101]
df_101_2016=df_2016[df_2016['freeway']==101]
df_101_2017=df_2017[df_2017['freeway']==101]
df_101_2018=df_2018[df_2018['freeway']==101]
df_101_2019=df_2019[df_2019['freeway']==101]
df_101_2020=df_2020[df_2020['freeway']==101]


# ## <font color=blue> Filtering the 400001 Station id alone

# In[33]:

df_week_2015=df_101_2015[df_101_2015['station']==400001]
df_week_2016=df_101_2016[df_101_2016['station']==400001]
df_week_2017=df_101_2017[df_101_2017['station']==400001]
df_week_2018=df_101_2018[df_101_2018['station']==400001]
df_week_2019=df_101_2019[df_101_2019['station']==400001]
df_week_2020=df_101_2020[df_101_2020['station']==400001]


# ## <font color=blue>Data Preparation

# In[34]:

print(df_week_2015.shape)
print(df_week_2016.shape)
print(df_week_2017.shape)
print(df_week_2018.shape)
print(df_week_2019.shape)
print(df_week_2020.shape)


# ## <font color=blue> Removing the duplicates

# In[35]:

df1_2015=df_week_2015.drop_duplicates(subset=['station','timestamp_'])
df1_2016=df_week_2016.drop_duplicates(subset=['station','timestamp_'])
df1_2017=df_week_2017.drop_duplicates(subset=['station','timestamp_'])
df1_2018=df_week_2018.drop_duplicates(subset=['station','timestamp_'])
df1_2019=df_week_2019.drop_duplicates(subset=['station','timestamp_'])
df1_2020=df_week_2020.drop_duplicates(subset=['station','timestamp_'])


# In[36]:

df1_2015['day_of_week_num'].value_counts()


# In[37]:

cols = ['station','timestamp_','occupancy','day_of_week_num','hour_of_day','speed']
df1_2015= (df1_2015[cols].set_index(['station','timestamp_']).sort_values(['station','timestamp_']))
df1_2016= (df1_2016[cols].set_index(['station','timestamp_']).sort_values(['station','timestamp_']))
df1_2017= (df1_2017[cols].set_index(['station','timestamp_']).sort_values(['station','timestamp_']))
df1_2018= (df1_2018[cols].set_index(['station','timestamp_']).sort_values(['station','timestamp_']))
df1_2019= (df1_2019[cols].set_index(['station','timestamp_']).sort_values(['station','timestamp_']))
df1_2020= (df1_2020[cols].set_index(['station','timestamp_']).sort_values(['station','timestamp_']))



# In[38]:

df1_2019.head()


# In[39]:

from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import numpy as np
import math
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[40]:

def data_index(df, n_lag, n_steps):
    
    steps = n_lag + n_steps - 1
    
    return df.iloc[steps:].index

def format_model_data(df, n_lag, n_steps):
    df_out = []
    for station, new_df in df.groupby(level=0):
        key, scaled, scaler1 = scale_data(new_df)
        reframed_ = prepare_data_for_network(scaled,n_lag,n_steps)
        df_out.append(reframed_)
        
    return pd.concat(df_out, ignore_index=True), key, scaled, scaler1


def scale_data(df):
    # process data
    values = df.values
    key = data_key(df)

    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler1 = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)

    scaled1 = scaler1.fit_transform(values[:,3].reshape(-1, 1))
    # print(key)
    
    return key, scaled, scaler1


def data_key(df):
    key = dict()
    for i,col in enumerate(list(df.columns)):
        var_ = 'var'+str(i+1)
        key[col] = var_
    return key

def prepare_data_for_network(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

#def remove_cols(df):
    
    #cols = list(df.columns)
    #drop_1 = [c for c in cols if '(t+' in c]
    #drop_2 = [c for c in cols if '(t)' in c]

    #drop_1.remove('var4(t+5)')

    #drop_cols = drop_1 + drop_2
    #df = df.drop(drop_cols, axis=1)
    
    #return df


# In[19]:

from sklearn import metrics
def predict_data(df, model, scaler1):
    
    test_X, test_y = prepare_data(df)
    
    my_model = tf.keras.models.load_model(model)
    
    # make a prediction
    yhat = my_model.predict(test_X)
    print('yhat predicted')
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler1.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]

    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler1.inverse_transform(inv_y)
    inv_y = inv_y[:,0]

    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    
    mse=mean_squared_error(inv_y, inv_yhat)
    print('Test MSE: %.3f' % mse)
    
    mae=mean_absolute_error(inv_y,inv_yhat)
    print('Test MAE: %.3f' % mae)
    
    mape=mean_absolute_percentage_error(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    accuracy=metrics.r2_score(inv_y,inv_yhat)
    print('R^2:%.3f'% accuracy)
    
    return inv_y, inv_yhat, rmse,mse,mae,mape,accuracy


    
    

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def prepare_data(df):
    test = df.values
    
    # split into input and outputs
    test_X, test_y = test[:, :-1], test[:, -1]
    
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    
    print(test_X.shape, test_y.shape)
    
    return test_X, test_y


# ## <font color=blue>Training the model with 2015,2016,2017,2018,2019 and testing for 2020

# In[41]:

frames=[df1_2015,df1_2016,df1_2017,df1_2018,df1_2019]
df_five=pd.concat(frames)
df_five.head()


# In[42]:

df_five.tail(10)


# In[43]:

#define how many timesteps to look back as input with the variable n_lag.
n_lag = 3

#define how many timesteps ahead to predict with the variable n_steps.
n_steps = 1
predict_col = 'speed'
reframed, key, scaled, scaler1 =format_model_data(df_five, n_lag, n_steps)

reframed.drop(reframed.columns[[12,13,14]], axis=1, inplace=True)

#reframed contains the data in the correct format for the model
reframed.head()


# In[44]:

# define split
train_ratio = 0.6
val_ratio = 0.2

train_val = int(reframed.shape[0] * train_ratio)

val_test = train_val + int(reframed.shape[0] * val_ratio)

print("Size of training set:", train_val)
print("Size of Validation set:", val_test-train_val)
print("Size of Testing set:", reframed.shape[0]-val_test)
#define number of steps in to the future

print(reframed.shape)


# In[45]:

#Data
values = reframed.values
train = values[:train_val, :]
val = values[train_val:val_test, :]
test = values[val_test:, :]


# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
val_X, val_y = val[:, :-1], val[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[46]:

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM,GRU, CuDNNLSTM, Activation


# In[47]:

traffic_model5 = Sequential()
traffic_model5.add(LSTM(input_shape=(train_X.shape[1], train_X.shape[2]), units=32, return_sequences=True))
#traffic_model5.add(Dropout(0.3))
traffic_model5.add(LSTM(units=32, return_sequences=False))
#traffic_model5.add(Dropout(0.2))

traffic_model5.add(Dense(units=1))
traffic_model5.add(Activation("sigmoid"))
traffic_model5.compile(loss='mse', optimizer='adam')


# In[48]:

traffic_prediction_2015=traffic_model5.fit(train_X,train_y,epochs=50,batch_size=32,validation_data=(val_X,val_y),verbose=2,shuffle=False)


# In[49]:

plt.figure(figsize = (8,5))
plt.plot(traffic_prediction_2015.history['loss'],'o-', color="r", label='train loss')
plt.plot(traffic_prediction_2015.history['val_loss'],'o-', color="g" ,label='validation loss')
plt.legend()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Loss', fontsize = 20)
plt.xlabel('Epoch', fontsize = 20)
plt.title('Loss Curve for LSTM Model - 400001 Station', fontsize = 18, y = 1.03)
    
plt.show()


# In[ ]:

# make a prediction
yhat = traffic_model5.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler1.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler1.inverse_transform(inv_y)
inv_y = inv_y[:,0]

import math
# calculate RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[37]:

#Save partly trained model
traffic_model5.save('trained2015_best.h5')


# In[ ]:




# ## <font color=blue> Reloading the data and predicting for 2020

# In[38]:

#define how many timesteps to look back as input with the variable n_lag.
n_lag = 3

#define how many timesteps ahead to predict with the variable n_steps.
n_steps = 1
predict_col = 'speed'

treframed, tkey, tscaled, tscaler1 = format_model_data(df1_2020, n_lag, n_steps)
treframed.drop(treframed.columns[[12,13,14]], axis=1, inplace=True)

treframed.head()


# In[40]:

tinv_y, tinv_yhat, trmse,tmse,tmae,tmape,accuracy= predict_data(treframed,'./trained2015_best.h5',tscaler1)


# In[ ]:

treframed.shape


# In[ ]:

import tensorflow as tf
model2019=tf.keras.models.load_model('./trained2015.h5')


# In[ ]:

model2019.summary()


# In[41]:

tinv_y, tinv_yhat, trmse,tmse,tmae,tmape,accuracy= predict_data(treframed,'./trained2015_best.h5',tscaler1)


# In[42]:

from matplotlib.pyplot import figure

figure(num=None, figsize=(18, 6), dpi=80, edgecolor='k')
plt.title("Traffic Speed LSTM Model ",fontsize=20)
plt.plot(tinv_y[0:24],'o-', linewidth=4,color="midnightblue",label='Actual');
plt.plot(tinv_yhat[1:25],'o-',linewidth=4, color="orangered",label='Predicted');
plt.ylabel('Speed', fontsize = 18)
plt.xlabel('Training size', fontsize = 18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.show()


# In[43]:

from matplotlib.pyplot import figure

figure(num=None, figsize=(18, 6), dpi=80, edgecolor='k')
plt.title("Prediction using Traffic Data",fontsize=18)
plt.plot(tinv_y[0:100],'o-',color='red',label='Actual');
plt.plot(tinv_yhat[1:101],'o-',color='green',label='Predicted');
plt.ylabel('Speed', fontsize = 14)
plt.xlabel('Training size', fontsize = 14)
plt.legend()


# In[44]:

y_actual=pd.DataFrame(tinv_y)
y_predicted=pd.DataFrame((tinv_yhat))
df_2020_s=pd.concat([y_actual,y_predicted],axis=1)
col=['y_actual','y_predicted']
df_2020_s.columns=col
#df_2020.columns=['y_actual','y_predicted']
df_2020_s['y_predicted']=df_2020_s['y_predicted'].round(1)

df_2020_s.reset_index()
df_2020_s.head(20)


# ## <font color=blue> Have to skip first row from y_predicted as it is predicting 1 hour head

# In[45]:

y1_actual=y_actual[:-1]
y1_actual.reset_index(inplace=True,drop=True)
#y1_actual.head()

y1_predicted=y_predicted[1:]
y1_predicted.reset_index(inplace=True,drop=True)
y1_predicted.head()


# In[47]:

y1_predicted[0]=y1_predicted[0].round(1)
y1_predicted.head()


# In[49]:

df_2020_s1=pd.concat([y1_actual,y1_predicted],axis=1)


col=['y_actual','y_predicted']
df_2020_s1.columns=col

df_2020_s1.head()


# In[50]:

data=df1_2020.reset_index()
data.head(20)


# In[51]:

data1=data.tail(-3)
data1.reset_index(inplace = True, drop = True) 
data1.head()


# In[52]:

final_1=pd.concat([data1,df_2020_s1],axis=1)
final_1.head(20)


# In[53]:

final_1.head(15)


# In[54]:

final_1.tail(10)


# In[142]:

final_1.head(50)


# In[56]:

final_1.to_csv("LSTM.csv")


# In[55]:

from matplotlib.pyplot import figure
import matplotlib.dates as mdates

figure(num=None, figsize=(14, 6), dpi=80, edgecolor='k')
plt.title("LSTM Model Traffic Speed Prediction",fontsize=18)
plt.plot(final_1['timestamp_'][:50],final_1['y_actual'][:50],'o-',color='darkgreen',label='Actual');
plt.plot(final_1['timestamp_'][:50],final_1['y_predicted'][:50],'o-',color='darkorange',label='Predicted');
plt.ylabel('Speed', fontsize = 20)
plt.xlabel('Date', fontsize = 20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gcf().autofmt_xdate()
ax = plt.gca()
xfmt = mdates.DateFormatter('%d-%m-%Y %H:%M')
ax.xaxis.set_major_formatter(xfmt)
plt.legend(loc='best')


# In[ ]:




# ## <font color=blue> GRU

# In[ ]:




# In[58]:

from tensorflow.python.keras import optimizers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM,GRU, CuDNNLSTM, Activation


# In[59]:

model_GRU = Sequential()
model_GRU.add(GRU(32, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
model_GRU.add(GRU(32))
model_GRU.add(Dropout(0.1))
model_GRU.add(Dense(1))
model_GRU.add(Activation("sigmoid"))
model_GRU.compile(loss='mse', optimizer='adam')


# In[60]:

traffic_prediction_GRU=model_GRU.fit(train_X,train_y,epochs=50,batch_size=100,validation_data=(val_X,val_y),verbose=2)


# In[61]:


# plot history
plt.figure(figsize = (8,5))
plt.plot(traffic_prediction_GRU.history['loss'],'o-', color="r", label='train')
plt.plot(traffic_prediction_GRU.history['val_loss'],'o-', color="g" ,label='val')
plt.legend()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel('Loss', fontsize = 18)
plt.xlabel('Epoch', fontsize = 18)
plt.title('Loss Curve for GRU Model - 400001 station', fontsize = 20, y = 1.03)
    

plt.legend()
plt.show()


# In[62]:

#Data
values = reframed.values
train = values[:train_val, :]
val = values[train_val:val_test, :]
test = values[val_test:, :]


# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
val_X, val_y = val[:, :-1], val[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[63]:

# make a prediction
yhat = model_GRU.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


# invert scaling for forecast
inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler1.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler1.inverse_transform(inv_y)
inv_y = inv_y[:,0]

import math
# calculate RMSE
rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


# In[64]:

#Save partly trained model
model_GRU.save('trained_GRU2015.h5')


# In[65]:


model_GRU.save('trained2015_GRU.h5')


# In[66]:

#define how many timesteps to look back as input with the variable n_lag.
n_lag = 3

#define how many timesteps ahead to predict with the variable n_steps.
n_steps = 1
predict_col = 'speed'

treframed, tkey, tscaled, tscaler1 = format_model_data(df1_2020, n_lag, n_steps)
treframed.drop(treframed.columns[[12,13,14]], axis=1, inplace=True)

treframed.head()


# In[67]:

tinv_y, tinv_yhat, trmse,tmse,tmae,tmape,accuracy= predict_data(treframed,'./trained2015_GRU.h5',tscaler1)


# In[68]:

import tensorflow as tf
model2019_GRU=tf.keras.models.load_model('./trained2015_GRU.h5')


# In[69]:

model2019_GRU.summary()


# In[70]:

from matplotlib.pyplot import figure

figure(num=None, figsize=(20, 6), dpi=80, edgecolor='k')
plt.title("Prediction using Traffic Data")
plt.plot(tinv_y[0:100],label='Actual');
plt.plot(tinv_yhat[1:101],label='Predicted');
plt.legend()


# In[71]:

from matplotlib.pyplot import figure

figure(num=None, figsize=(18, 6), dpi=80, edgecolor='k')
plt.title("Traffic Speed GRU Model",fontsize=20)
plt.plot(tinv_y[0:24],'*-',linewidth=4,color='midnightblue',label='Actual');
plt.plot(tinv_yhat[1:25],'o-',linewidth=4,color='orangered',label='Predicted');
plt.ylabel('Speed', fontsize = 18)
plt.xlabel('Training size', fontsize = 18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()
plt.legend()


# In[72]:

y_actual=pd.DataFrame(tinv_y)
y_predicted=pd.DataFrame((tinv_yhat))
df_2020_s=pd.concat([y_actual,y_predicted],axis=1)
col=['y_actual','y_predicted']
df_2020_s.columns=col
#df_2020.columns=['y_actual','y_predicted']
df_2020_s['y_predicted']=df_2020_s['y_predicted'].round(1)

df_2020_s.reset_index()
df_2020_s.head(20)


# In[73]:

y1_actual=y_actual[:-1]
y1_actual.reset_index(inplace=True,drop=True)
#y1_actual.head()

y1_predicted=y_predicted[1:]
y1_predicted.reset_index(inplace=True,drop=True)
y1_predicted.head()


# In[74]:

y1_predicted[0]=y1_predicted[0].round(1)
y1_predicted.head()




# In[75]:

df_2020_s1=pd.concat([y1_actual,y1_predicted],axis=1)


col=['y_actual','y_predicted']
df_2020_s1.columns=col

df_2020_s1.head()


# In[76]:

data=df1_2020.reset_index()
data.head(20)


# In[77]:

data1=data.tail(-3)
data1.reset_index(inplace = True, drop = True) 
data1.head()


# In[78]:

final_2=pd.concat([data1,df_2020_s1],axis=1)
final_2.head(20)


# In[79]:

final_2.to_csv("GRU.csv")


# In[80]:

from matplotlib.pyplot import figure
import matplotlib.dates as mdates

figure(num=None, figsize=(14, 6), dpi=80, edgecolor='k')
plt.title("GRU Model Traffic Speed Prediction",fontsize=18)
plt.plot(final_2['timestamp_'][:50],final_2['y_actual'][:50],'o-',color='darkgreen',label='Actual');
plt.plot(final_2['timestamp_'][:50],final_2['y_predicted'][:50],'o-',color='darkorange',label='Predicted');
plt.ylabel('Speed', fontsize = 20)
plt.xlabel('Date', fontsize = 20)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.gcf().autofmt_xdate()
ax = plt.gca()
xfmt = mdates.DateFormatter('%d-%m-%Y %H:%M')
ax.xaxis.set_major_formatter(xfmt)
plt.legend(loc='best')


# In[ ]:



