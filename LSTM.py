import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from numpy.random import seed

# 统一使用TensorFlow Keras API
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam

# 自定义模块
from utils import *
from model import lstm  # 确保model.py已更新

# GPU配置
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices([gpus[0]], "GPU")
        print("GPU configured successfully")
    except RuntimeError as e:
        print(e)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# 设置随机种子
seed(1)
tf.random.set_seed(1)

# 参数设置
n_timestamp = 10
n_epochs = 50
model_type = 3  # bidirectional LSTM

# 数据加载
yuan_data = pd.read_csv('/mnt/disk2/peiling.chen/Attention-CLX-stock-prediction/601988.SH.csv')  
yuan_data.index = pd.to_datetime(yuan_data['trade_date'], format='%Y%m%d') 
yuan_data = yuan_data.loc[:, ['open', 'high', 'low', 'close', 'amount']]

data = pd.read_csv('./ARIMA_residuals1.csv')
data.index = pd.to_datetime(data['trade_date'])
data = data.drop('trade_date', axis=1)

# 数据分割
Lt = pd.read_csv('./ARIMA.csv')
idx = 3500
training_set = data.iloc[1:idx, :]
test_set = data.iloc[idx:, :]
yuan_training_set = yuan_data.iloc[1:idx, :]
yuan_test_set = yuan_data.iloc[idx:, :]

# 数据标准化
sc = MinMaxScaler(feature_range=(0, 1))
yuan_sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.transform(test_set)  # 注意这里使用transform而不是fit_transform
yuan_training_set_scaled = yuan_sc.fit_transform(yuan_training_set)
yuan_testing_set_scaled = yuan_sc.transform(yuan_test_set)  # 同上

# 数据准备
def data_split(data, n_steps):
    X, y = [], []
    for i in range(len(data)-n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

X_train, y_train = data_split(training_set_scaled, n_timestamp)
yuan_X_train, yuan_y_train = data_split(yuan_training_set_scaled, n_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
yuan_X_train = yuan_X_train.reshape(yuan_X_train.shape[0], yuan_X_train.shape[1], 5)

X_test, y_test = data_split(testing_set_scaled, n_timestamp)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
yuan_X_test, yuan_y_test = data_split(yuan_testing_set_scaled, n_timestamp)
yuan_X_test = yuan_X_test.reshape(yuan_X_test.shape[0], yuan_X_test.shape[1], 5)  # 修正变量名拼写错误

# 模型构建和训练
model, yuan_model = lstm(model_type, X_train, yuan_X_train)
print(model.summary())

adam = Adam(learning_rate=0.01)
model.compile(optimizer=adam, loss='mse')
yuan_model.compile(optimizer=adam, loss='mse')

history = model.fit(X_train, y_train,
                   batch_size=32,
                   epochs=n_epochs,
                   validation_data=(X_test, y_test),
                   validation_freq=1)

yuan_history = yuan_model.fit(yuan_X_train, yuan_y_train,
                            batch_size=32,
                            epochs=n_epochs,
                            validation_data=(yuan_X_test, yuan_y_test),
                            validation_freq=1)

# 绘制损失曲线
def plot_history(history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.legend()
    plt.show()

plot_history(history, 'residuals: Training and Validation Loss')
plot_history(yuan_history, 'LSTM: Training and Validation Loss')

# 预测和结果可视化
yuan_predicted_stock_price = yuan_model.predict(yuan_X_test)
yuan_predicted_stock_price = yuan_sc.inverse_transform(yuan_predicted_stock_price)
yuan_predicted_stock_price_list = yuan_predicted_stock_price[:, 3].flatten()

yuan_real_stock_price = yuan_sc.inverse_transform(yuan_y_test)
yuan_real_stock_price_list = yuan_real_stock_price[:, 3].flatten()

predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
predicted_stock_price_list = predicted_stock_price.flatten()

# 创建DataFrame
def create_df(index, values, column='close'):
    return pd.DataFrame({column: values}, index=index)

yuan_predicted_df = create_df(yuan_data.index[idx+n_timestamp:], yuan_predicted_stock_price_list)
yuan_real_df = create_df(yuan_data.index[idx+n_timestamp:], yuan_real_stock_price_list)
predicted_df = create_df(data.index[idx+n_timestamp:], predicted_stock_price_list)

# 合并预测结果
final_predicted = pd.concat([Lt.set_index('trade_date'), predicted_df]).groupby('trade_date')['close'].sum().reset_index()
final_predicted.index = pd.to_datetime(final_predicted['trade_date'])
final_predicted = final_predicted.drop('trade_date', axis=1)

# 绘制预测结果
def plot_prediction(real, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(real, label='Actual Price')
    plt.plot(predicted, label='Predicted Price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

plot_prediction(yuan_data.loc['2021-06-22':, 'close'], 
               final_predicted['close'], 
               'BiLSTM: Stock Price Prediction')

plot_prediction(yuan_real_df['close'], 
               yuan_predicted_df['close'], 
               'LSTM: Stock Price Prediction')

# 评估指标
evaluation_metric(final_predicted['close'], yuan_data.loc['2021-06-22':, 'close'])