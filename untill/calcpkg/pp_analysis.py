from datetime import datetime 
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Activation,LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def split_sequences(sequence, n_steps):
    X,y = list(), list()
    for i in range(len(sequence)):
        end_ix = i+n_steps
        if end_ix > len(sequence)-1:
            break            
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def data_setup(n_steps, n_seq, sequence):
    X, y = split_sequences(sequence, n_steps)
    n_features = X.shape[2]
    X = X.reshape((len(X), n_steps, n_features))
    new_y = []
    for term in y:
        new_term = term[-1]
        new_y.append(new_term)
    return X, np.array(new_y), n_features

# 데이터 불러오기
pp_df=pd.read_csv('/data1/CHH/현대모비스/정형데이터/weekly_pp_final.csv',encoding='cp949')

# 데이터 프레임 및 하이퍼 파라미터 
pp_list=[(pp_df, 2, 'sigmoid','linear', 'relu', 'sigmoid',30,50,10,10,'adam',0.07,10,4),
(pp_df, 4, 'sigmoid',None, 'relu', 'sigmoid',30,50,10,10,'adam',0.07,10,4),
(pp_df, 7, 'sigmoid',None, 'relu', 'sigmoid',30,50,10,10,'adam',0.07,10,4)]
    # (2,pp_df_word, 256,30, 'relu','relu','sgd'),
    # (4,pp_df_word, 256,30, 'relu','relu','sgd'),
    # (7,pp_df_word, 256,30, 'relu','relu','sgd')]

# def pp_analysis():
    # path
    # data
    # parameter 설정 
    #change_df =원자재 데이터, month_s= 예측 개월수 , lstm_1 = LSTM_1 활성화층, lstm_2 = LSTM_1 활성화층, 
    # den_1= dense_1_활성화층, den_2= dense_2_활성화층, lstm_num1 = LSTM_1_층 개수, 
    # lstm_num2 = LSTM_2_층 개수,  den_num1 = Dense_1_층 개수, den_num2=Dense_2_층 개수, 
    # opti = optimizer 방법, learning_rate_num= 학습률, batch_num= 배치사이즈 수, nstep= 윈도우 사이즈 수
result_save=pd.DataFrame()
val_mape_list=[]
for change_df, month_s, lstm_1,lstm_2,den_1,den_2, lstm_num1, lstm_num2,  den_num1, den_num2, opti, learning_rate_num, batch_num, nstep in pp_list:

    change_df.reset_index(drop=True,inplace=True)
    dat=change_df.copy()

    yy=dat[dat.columns[1]].shift((month_s-1)*-4)
    feature_x=dat[dat.columns[1:]]
    yy.dropna(inplace=True)
    feature_x.reset_index(inplace=True,drop=True)
    yy.reset_index(inplace=True,drop=True)
    dat=pd.concat([feature_x[:len(yy)],yy],axis=1)

    train_x=dat.iloc[:-(month_s*4)-(nstep-4),:-1]
    train_y=dat.iloc[:-(month_s*4)-(nstep-4),-1]
    vpp_x=dat.iloc[-(month_s*4)-(nstep-4):,:-1]
    vpp_y=dat.iloc[-(month_s*4)-(nstep-4):,-1]
    test_x=change_df.iloc[-(month_s*4)-(nstep-4):,1:]
    test_y=change_df.iloc[-(month_s*4)-(nstep-4):,1]

    train_x.reset_index(inplace=True,drop=True)
    train_y.reset_index(inplace=True,drop=True)
    test_x.reset_index(inplace=True,drop=True)
    test_y.reset_index(inplace=True,drop=True)
    vpp_x.reset_index(inplace=True,drop=True)
    vpp_y.reset_index(inplace=True,drop=True)

    vpp_total=pd.concat([vpp_x,vpp_y],axis=1)
    train_total=pd.concat([train_x,train_y],axis=1)
    test_total=pd.concat([test_x,test_y],axis=1)

    dff=train_total.values
    dff1=test_total.values
    dff2=vpp_total.values

    #스케일러 과정
    scaler = MinMaxScaler()
    scaler.fit(dff)
    dff = scaler.transform(dff)
    mvalues = dff
    dff1 = scaler.transform(dff1)
    dvalues = dff1
    dff2 = scaler.transform(dff2)
    vvalues = dff2

    n_steps = nstep
    n_seq = 30

    X, y, n_reatures = data_setup(n_steps, n_seq, mvalues)
    X = X[:,:,:-1]
    y = y[:]
    X_train, y_train = X[:int(len(X))], y[:int(len(X))]

    vX, vy, vn_reatures = data_setup(n_steps, n_seq, vvalues)
    vX = vX[:,:,:-1]
    vy = vy[:]
    X_val, y_val = vX[:int(len(vX))], vy[:int(len(vX))]

    dX, dy, dn_reatures = data_setup(n_steps, n_seq, dvalues)
    dX = dX[:,:,:-1]
    dy = dy[:]
    X_test, y_test = dX[:int(len(dX))], dy[:int(len(dX))]

    learning_rate = learning_rate_num
    training_cnt = 300

    # 모델 구조 설정
    tf.random.set_seed(42)
    model = Sequential()
    model.add(LSTM(lstm_num1 , activation =lstm_1,input_shape = (X.shape[1], X.shape[2]), return_sequences = True))
    model.add(LSTM(lstm_num2 , activation=lstm_2, return_sequences=True))
    model.add(Dense(den_num1, activation=den_1))
    model.add(Dense(den_num2, activation=den_2))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer=opti, metrics=['mape','mse'])
    callback_list = [EarlyStopping(monitor='val_mape', mode='min', verbose=0, patience=15,restore_best_weights=True)]
    epochs = 40
    verbosity = 0
    history =model.fit(X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_num,
            validation_data = (X_val, y_val),
            verbose = verbosity,
        callbacks = callback_list)
    hist_df = pd.DataFrame(history.history)
    pred= model.predict(X_test)    
    y_all_predict=pred.mean(axis=1).reshape(-1,1)
    price_pred=np.concatenate((dff[:4*(month_s-1),:dff.shape[1]-1],y_all_predict),axis=1)
    result_pred=scaler.inverse_transform(price_pred)
    result_pred=result_pred[:,-1]
    result_pred=pd.DataFrame(result_pred)
    result_save=pd.concat([result_save,result_pred],axis=1)
    val_mape_list.append(hist_df['val_mape'].values[-16])

result_save.columns=['1_month','3_month','6_month']
# result_save.to_csv('./Result/pp_predict_values.csv',encoding='utf-8-sig',index=False)
pp = result_save.copy()
pp_mape=val_mape_list
# pp_analysis()
print('Done!')