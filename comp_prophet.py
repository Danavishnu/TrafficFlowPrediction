import pandas as pd
import fbprophet as Prophet
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_prophet(train,test):
    df1 = pd.read_csv(train, encoding='utf-8', usecols=["5 Minutes","Lane 1 Flow (Veh/5 Minutes)"]).fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8',usecols=["5 Minutes","Lane 1 Flow (Veh/5 Minutes)"]).fillna(0)
    df1 = df1.rename(columns = {'5 Minutes':'ds','Lane 1 Flow (Veh/5 Minutes)':'y'})
    df2 = df2.rename(columns = {'5 Minutes':'ds','Lane 1 Flow (Veh/5 Minutes)':'y'})
    df2 = df2[12:]
    # df1=X_train
    # df2=Y_train
    df1['ds'] = pd.to_datetime(df1['ds'])
    df2['ds'] = pd.to_datetime(df2['ds'])
    scaler= MinMaxScaler(feature_range=(0, 1)).fit(df1['y'].values.reshape(-1, 1))
    df1['y'] = scaler.transform(df1['y'].values.reshape(-1, 1)).reshape(1, -1)[0]

    print(df1)
    model = Prophet.Prophet(changepoint_prior_scale=0.01)
    start = time.time()
    model.fit(df1)
    end = time.time()
    print('Time Required :', end-start,'seconds')
    print("Predicting...")
    predicted=(model.predict(df2.iloc[:,0:1])['yhat']).to_numpy()
    predicted=scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(1, -1)[0]
    print(predicted)
    # predicted = (model.predict(df2.iloc[:,0:1])['yhat']).to_numpy()
    # print(df2['y'])
    # print(predicted)
    # print(type(predicted))
    # print(predicted.shape)
    return predicted

# get_prophet("data/train.csv","data/test.csv")