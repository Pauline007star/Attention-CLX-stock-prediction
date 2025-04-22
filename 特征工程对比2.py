import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制使用CPU
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
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.optimizers import Adam

# GPU配置
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices([gpus[0]], "GPU")
        print("GPU configured successfully")
    except RuntimeError as e:
        print(e)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 设置随机种子
seed(1)
tf.random.set_seed(1)

# 参数设置
n_timestamp = 10
n_epochs = 50
model_type = 3  # bidirectional LSTM

# ==================== 数据加载 ====================
# 1. 原始股票数据
yuan_data = pd.read_csv('/mnt/disk2/peiling.chen/Attention-CLX-stock-prediction/601988.SH.csv')  
yuan_data.index = pd.to_datetime(yuan_data['trade_date'], format='%Y%m%d') 
yuan_data = yuan_data.loc[:, ['open', 'high', 'low', 'close', 'amount']]

# 2. ARIMA残差数据
data = pd.read_csv('/mnt/disk2/peiling.chen/Attention-CLX-stock-prediction/ARIMA_residuals1.csv')
data.index = pd.to_datetime(data['trade_date'])
data = data.drop('trade_date', axis=1)

# 3. 特征工程数据
filtered_alpha_features = pd.read_csv('/mnt/disk2/peiling.chen/finbot/finbot/modules/forecast/alpha158_features_PCA.csv')
filtered_alpha_features.index = pd.to_datetime(filtered_alpha_features['datetime'])
filtered_alpha_features = filtered_alpha_features.drop('datetime', axis=1)

# 4. 新增的merged数据
merged_data = pd.read_csv('/mnt/disk2/peiling.chen/Attention-CLX-stock-prediction/merged_data_PCA.csv')
merged_data.index = pd.to_datetime(merged_data['datetime'], format='%Y-%m-%d')
merged_data = merged_data.drop('datetime', axis=1)

# ARIMA基础预测
Lt = pd.read_csv('/mnt/disk2/peiling.chen/Attention-CLX-stock-prediction/ARIMA.csv')
Lt.index = pd.to_datetime(Lt['trade_date'])
Lt = Lt.drop('trade_date', axis=1)

# ==================== 数据分割 ====================
idx = 3500

# 打印原始数据集的长度
print("Length of yuan_data:", len(yuan_data))
print("Length of data (ARIMA residuals):", len(data))
print("Length of filtered_alpha_features:", len(filtered_alpha_features))
print("Length of merged_data:", len(merged_data))

# 数据分割
yuan_training_set = yuan_data.iloc[1:idx, :]
yuan_test_set = yuan_data.iloc[idx:, :]

training_set = data.iloc[1:idx, :]
test_set = data.iloc[idx:, :]

alpha_training_set = filtered_alpha_features.iloc[1:idx, :]
alpha_test_set = filtered_alpha_features.iloc[idx:, :]

merged_training_set = merged_data.iloc[1:idx, :]
merged_test_set = merged_data.iloc[idx:, :]

# 打印分割后的数据集长度
print("Length of yuan_training_set:", len(yuan_training_set))
print("Length of yuan_test_set:", len(yuan_test_set))
print("Length of training_set:", len(training_set))
print("Length of test_set:", len(test_set))
print("Length of alpha_training_set:", len(alpha_training_set))
print("Length of alpha_test_set:", len(alpha_test_set))
print("Length of merged_training_set:", len(merged_training_set))
print("Length of merged_test_set:", len(merged_test_set))

# ==================== 数据标准化 ====================
# 原始数据
yuan_sc = MinMaxScaler(feature_range=(0, 1))
yuan_training_set_scaled = yuan_sc.fit_transform(yuan_training_set)
yuan_testing_set_scaled = yuan_sc.transform(yuan_test_set)

# ARIMA残差
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.transform(test_set)

# 特征工程数据
alpha_sc = MinMaxScaler(feature_range=(0, 1))
alpha_training_set_scaled = alpha_sc.fit_transform(alpha_training_set)
alpha_testing_set_scaled = alpha_sc.transform(alpha_test_set)

# Merged数据
merged_sc = MinMaxScaler(feature_range=(0, 1))
merged_training_set_scaled = merged_sc.fit_transform(merged_training_set)
merged_testing_set_scaled = merged_sc.transform(merged_test_set)

# 打印标准化后的数据集长度
print("Length of yuan_training_set_scaled:", len(yuan_training_set_scaled))
print("Length of yuan_testing_set_scaled:", len(yuan_testing_set_scaled))
print("Length of training_set_scaled:", len(training_set_scaled))
print("Length of testing_set_scaled:", len(testing_set_scaled))
print("Length of alpha_training_set_scaled:", len(alpha_training_set_scaled))
print("Length of alpha_testing_set_scaled:", len(alpha_testing_set_scaled))
print("Length of merged_training_set_scaled:", len(merged_training_set_scaled))
print("Length of merged_testing_set_scaled:", len(merged_testing_set_scaled))

# ==================== 数据准备 ====================
def data_split(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i + n_steps])
        y.append(data[i + n_steps])
    return np.array(X), np.array(y)

# 原始数据
yuan_X_train, yuan_y_train = data_split(yuan_training_set_scaled, n_timestamp)
yuan_X_train = yuan_X_train.reshape(yuan_X_train.shape[0], yuan_X_train.shape[1], 5)
yuan_X_test, yuan_y_test = data_split(yuan_testing_set_scaled, n_timestamp)
yuan_X_test = yuan_X_test.reshape(yuan_X_test.shape[0], yuan_X_test.shape[1], 5)

# ARIMA残差
X_train, y_train = data_split(training_set_scaled, n_timestamp)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test, y_test = data_split(testing_set_scaled, n_timestamp)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 特征工程数据
alpha_X_train, alpha_y_train = data_split(alpha_training_set_scaled, n_timestamp)
alpha_X_train = alpha_X_train.reshape(alpha_X_train.shape[0], alpha_X_train.shape[1], alpha_X_train.shape[2])
alpha_X_test, alpha_y_test = data_split(alpha_testing_set_scaled, n_timestamp)
alpha_X_test = alpha_X_test.reshape(alpha_X_test.shape[0], alpha_X_test.shape[1], alpha_X_train.shape[2])

# Merged数据
merged_X_train, merged_y_train = data_split(merged_training_set_scaled, n_timestamp)
merged_X_train = merged_X_train.reshape(merged_X_train.shape[0], merged_X_train.shape[1], merged_X_train.shape[2])
merged_X_test, merged_y_test = data_split(merged_testing_set_scaled, n_timestamp)
merged_X_test = merged_X_test.reshape(merged_X_test.shape[0], merged_X_test.shape[1], merged_X_train.shape[2])

# 打印最终的数据形状
print("Final shape of yuan_X_train:", yuan_X_train.shape)
print("Final shape of yuan_y_train:", yuan_y_train.shape)
print("Final shape of X_train:", X_train.shape)
print("Final shape of y_train:", y_train.shape)
print("Final shape of alpha_X_train:", alpha_X_train.shape)
print("Final shape of alpha_y_train:", alpha_y_train.shape)
print("Final shape of merged_X_train:", merged_X_train.shape)
print("Final shape of merged_y_train:", merged_y_train.shape)

# ==================== 模型构建 ====================
def lstm(model_type, X_train, yuan_X_train):
    # Original models
    if model_type == 1:
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(units=y_train.shape[1]))

        yuan_model = Sequential()
        yuan_model.add(LSTM(units=50, activation='relu', input_shape=(yuan_X_train.shape[1], yuan_X_train.shape[2])))
        yuan_model.add(Dense(units=yuan_y_train.shape[1]))

    elif model_type == 2:
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=50, activation='relu'))
        model.add(Dense(y_train.shape[1]))

        yuan_model = Sequential()
        yuan_model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(yuan_X_train.shape[1], yuan_X_train.shape[2])))
        yuan_model.add(LSTM(units=50, activation='relu'))
        yuan_model.add(Dense(yuan_y_train.shape[1]))

    elif model_type == 3:
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(y_train.shape[1]))

        yuan_model = Sequential()
        yuan_model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(yuan_X_train.shape[1], yuan_X_train.shape[2])))
        yuan_model.add(Dense(yuan_y_train.shape[1]))
    return model, yuan_model

def create_merged_model(merged_X_train):
    merged_model = Sequential()
    merged_model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(merged_X_train.shape[1], merged_X_train.shape[2])))
    merged_model.add(Dense(merged_y_train.shape[1]))
    return merged_model

def create_alpha_model(alpha_X_train):
    alpha_model = Sequential()
    alpha_model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(alpha_X_train.shape[1], alpha_X_train.shape[2])))
    alpha_model.add(Dense(alpha_y_train.shape[1]))
    return alpha_model

model, yuan_model = lstm(model_type, X_train, yuan_X_train)
merged_model = create_merged_model(merged_X_train)
alpha_model = create_alpha_model(alpha_X_train)

# 为每个模型创建独立的优化器实例
adam1 = Adam(learning_rate=0.01)
adam2 = Adam(learning_rate=0.01)
adam3 = Adam(learning_rate=0.01)
adam4 = Adam(learning_rate=0.01)

model.compile(optimizer=adam1, loss='mse')
yuan_model.compile(optimizer=adam2, loss='mse')
merged_model.compile(optimizer=adam3, loss='mse')
alpha_model.compile(optimizer=adam4, loss='mse')

# ==================== 模型训练 ====================
print("Training Yuan Data Model...")
yuan_history = yuan_model.fit(
    yuan_X_train, yuan_y_train,
    batch_size=32,
    epochs=n_epochs,
    validation_data=(yuan_X_test, yuan_y_test),
    validation_freq=1
)

print("\nTraining Residual Model...")
residual_history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=n_epochs,
    validation_data=(X_test, y_test),
    validation_freq=1
)

print("\nTraining Alpha Features Model...")
alpha_history = alpha_model.fit(
    alpha_X_train, alpha_y_train,
    batch_size=32,
    epochs=n_epochs,
    validation_data=(alpha_X_test, alpha_y_test),
    validation_freq=1
)

print("\nTraining Merged Data Model...")
merged_history = merged_model.fit(
    merged_X_train, merged_y_train,
    batch_size=32,
    epochs=n_epochs,
    validation_data=(merged_X_test, merged_y_test),
    validation_freq=1
)

# ==================== 绘制损失曲线 ====================
def plot_history(history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.legend()
    plt.show()

plot_history(yuan_history, 'Yuan Data: Training and Validation Loss')
plot_history(residual_history, 'Residuals: Training and Validation Loss')
plot_history(alpha_history, 'Alpha Features: Training and Validation Loss')
plot_history(merged_history, 'Merged Data: Training and Validation Loss')

# ==================== 预测和结果可视化 ====================
def create_df(index, values, column='close'):
    # 确保数据长度匹配
    min_len = min(len(index), len(values))
    return pd.DataFrame({column: values[:min_len]}, index=index[:min_len])

def plot_prediction(real, predicted, title):
    plt.figure(figsize=(10, 6))
    plt.plot(real, label='Actual Price')
    plt.plot(predicted, label='Predicted Price')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def evaluation_metric(predicted, real):
    # 确保数据长度一致
    min_len = min(len(predicted), len(real))
    predicted = predicted[:min_len]
    real = real[:min_len]
    
    mse = metrics.mean_squared_error(real, predicted)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(real, predicted)
    r2 = metrics.r2_score(real, predicted)

    print(f'MSE: {mse:.4f}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE: {mae:.4f}')
    print(f'R2 Score: {r2:.4f}')
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

# 1. 原始数据预测
yuan_predicted = yuan_model.predict(yuan_X_test)
yuan_predicted = yuan_sc.inverse_transform(yuan_predicted)
yuan_predicted_close = yuan_predicted[:, 3].flatten()
yuan_real = yuan_sc.inverse_transform(yuan_y_test)
yuan_real_close = yuan_real[:, 3].flatten()

# 确保索引长度匹配
yuan_test_index = yuan_data.index[idx+n_timestamp:idx+n_timestamp+len(yuan_predicted_close)]
yuan_predicted_df = create_df(yuan_test_index, yuan_predicted_close)
yuan_real_df = create_df(yuan_test_index, yuan_real_close)

# 2. ARIMA残差预测
residual_predicted = model.predict(X_test)
residual_predicted = sc.inverse_transform(residual_predicted)
residual_predicted_close = residual_predicted.flatten()

# 确保索引长度匹配
residual_test_index = data.index[idx+n_timestamp:idx+n_timestamp+len(residual_predicted_close)]
residual_predicted_df = create_df(residual_test_index, residual_predicted_close)

# 合并ARIMA预测结果
lt_values = Lt.loc[residual_test_index, 'close'].values
final_predicted = lt_values + residual_predicted_df['close'].values
final_predicted_df = create_df(residual_test_index, final_predicted)

# 3. 特征工程数据预测
alpha_predicted = alpha_model.predict(alpha_X_test)
alpha_predicted = alpha_sc.inverse_transform(alpha_predicted)
alpha_predicted_close = alpha_predicted.flatten()
alpha_real = alpha_sc.inverse_transform(alpha_y_test)
alpha_real_close = alpha_real.flatten()

# 确保索引长度匹配
alpha_test_index = filtered_alpha_features.index[idx+n_timestamp:idx+n_timestamp+len(alpha_predicted_close)]
alpha_predicted_df = create_df(alpha_test_index, alpha_predicted_close)
alpha_real_df = create_df(alpha_test_index, alpha_real_close)

# 4. Merged数据预测
merged_predicted = merged_model.predict(merged_X_test)
merged_predicted = merged_sc.inverse_transform(merged_predicted)
merged_predicted_close = merged_predicted.flatten()
merged_real = merged_sc.inverse_transform(merged_y_test)
merged_real_close = merged_real.flatten()

# 确保索引长度匹配
merged_test_index = merged_data.index[idx+n_timestamp:idx+n_timestamp+len(merged_predicted_close)]
merged_predicted_df = create_df(merged_test_index, merged_predicted_close)
merged_real_df = create_df(merged_test_index, merged_real_close)

# ==================== 结果可视化 ====================
# 绘制各模型的预测结果
plot_prediction(yuan_real_df['close'], yuan_predicted_df['close'], 
               'Yuan Data: Stock Price Prediction')

plot_prediction(yuan_data.loc[final_predicted_df.index, 'close'], 
               final_predicted_df['close'], 
               'ARIMA Residuals: Stock Price Prediction')

plot_prediction(alpha_real_df['close'], alpha_predicted_df['close'], 
               'Alpha Features: Stock Price Prediction')

plot_prediction(merged_real_df['close'], merged_predicted_df['close'], 
               'Merged Data: Stock Price Prediction')

# ==================== 模型评估 ====================
print("\n=== Model Evaluation ===")

# 收集所有评估结果
evaluation_results = {}

print("\n1. Yuan Data Model Evaluation:")
evaluation_results['Yuan'] = evaluation_metric(yuan_predicted_df['close'], yuan_real_df['close'])

print("\n2. ARIMA Residuals Model Evaluation:")
evaluation_results['ARIMA_Residuals'] = evaluation_metric(
    final_predicted_df['close'],
    yuan_data.loc[final_predicted_df.index, 'close']
)

print("\n3. Alpha Features Model Evaluation:")
evaluation_results['Alpha'] = evaluation_metric(alpha_predicted_df['close'], alpha_real_df['close'])

print("\n4. Merged Data Model Evaluation:")
evaluation_results['Merged'] = evaluation_metric(merged_predicted_df['close'], merged_real_df['close'])

# ==================== 对比分析 ====================
# 创建对比表格
metrics_df = pd.DataFrame.from_dict(evaluation_results, orient='index')
print("\n=== Performance Comparison ===")
print(metrics_df)

# 绘制指标对比图
plt.figure(figsize=(12, 8))
metrics_df.plot(kind='bar', subplots=True, layout=(2, 2), figsize=(14, 10))
plt.tight_layout()
plt.show()

# 绘制所有模型预测结果对比
plt.figure(figsize=(14, 8))
plt.plot(yuan_real_df.index, yuan_real_df['close'], label='Actual Price', linewidth=2)
plt.plot(yuan_predicted_df.index, yuan_predicted_df['close'], label='Yuan Data Prediction')
plt.plot(final_predicted_df.index, final_predicted_df['close'], label='ARIMA Residuals Prediction')
plt.plot(alpha_predicted_df.index, alpha_predicted_df['close'], label='Alpha Features Prediction')
plt.plot(merged_predicted_df.index, merged_predicted_df['close'], label='Merged Data Prediction')
plt.title('All Models Prediction Comparison')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()