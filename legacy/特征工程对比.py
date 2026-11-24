import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 减少TensorFlow日志输出
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from numpy.random import seed

from finbot.finutils.alpha import *



 
# 如果 model.py 中已经定义了 lstm() 函数就导入，否则下面提供一个示例定义
try:
    from model import lstm  # 确保 model.py 中有定义 lstm() 函数
except ImportError:
    # 示例 lstm() 函数定义，返回两个模型：一个用于残差数据，一个用于原始数据
    def lstm(model_type, X, Y):
        # 构建第一个模型：用于 ARIMA 残差数据
        if model_type == 3:
            model = Sequential()
            model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(X.shape[1], X.shape[2])))
            model.add(Dropout(0.2))
        else:
            model = Sequential()
            model.add(LSTM(64, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # 构建第二个模型：用于原始股票数据
        yuan_model = Sequential()
        yuan_model.add(LSTM(64, input_shape=(Y.shape[1], Y.shape[2])))
        yuan_model.add(Dense(1))
        yuan_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model, yuan_model

# GPU配置
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.set_visible_devices([gpus[0]], "GPU")
        print("GPU配置成功")
    except RuntimeError as e:
        print(e)

print("可用GPU数量: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 设置随机种子
seed(1)
tf.random.set_seed(1)

# 强制开启 eager execution 以解决 numpy() 调用问题
tf.config.run_functions_eagerly(True)

# 参数设置
n_timestamp = 10
n_epochs = 50
model_type = 3  # bidirectional LSTM

# ==================== 数据加载 ====================
print("\n=== 加载数据 ===")
# 1. 原始股票数据
yuan_data = pd.read_csv('/mnt/disk2/peiling.chen/Attention-CLX-stock-prediction/601988.SH.csv')
yuan_data.index = pd.to_datetime(yuan_data['trade_date'], format='%Y-%m-%d')
yuan_data = yuan_data.loc[:, ['open', 'high', 'low', 'close', 'amount']]

# 2. ARIMA残差数据
data = pd.read_csv('/mnt/disk2/peiling.chen/Attention-CLX-stock-prediction/ARIMA_residuals1.csv')
data.index = pd.to_datetime(data['trade_date'])
data = data.drop('trade_date', axis=1)

# 3. 特征工程数据
filtered_alpha_features = pd.read_csv('/mnt/disk2/peiling.chen/finbot/finbot/modules/forecast/filtered_alpha_features.csv')
filtered_alpha_features.index = pd.to_datetime(filtered_alpha_features['datetime'])
filtered_alpha_features = filtered_alpha_features.drop('datetime', axis=1)

# 4. 新增的merged数据
merged_data = pd.read_csv('/mnt/disk2/peiling.chen/Attention-CLX-stock-prediction/merged_data.csv')
merged_data.index = pd.to_datetime(merged_data['datetime'], format='%Y-%m-%d')
merged_data = merged_data.loc[:, [ 'change', 'vol', 'turnover_rate', 'volume_ratio', 'factor_x']]

# ==================== 数据预处理 ====================
# 删除非数值列
filtered_alpha_features = filtered_alpha_features.select_dtypes(include=['float64', 'int64'])

# 数据分割 - 使用统一的分割点
idx = 3500
total_samples = len(filtered_alpha_features)
print(f"\n总样本数: {total_samples}, 分割点: {idx}")

if idx >= total_samples:
    idx = int(total_samples * 0.8)
    print(f"警告: 分割点超出范围，自动调整为{idx}")

# 分割训练集和测试集
training_set = data.iloc[1:idx, :]
test_set = data.iloc[idx:, :]

yuan_training_set = yuan_data.iloc[1:idx, :]
yuan_test_set = yuan_data.iloc[idx:, :]

filtered_alpha_features_train = filtered_alpha_features.iloc[1:idx, :]
filtered_alpha_features_test = filtered_alpha_features.iloc[idx:, :]

merged_training_set = merged_data.iloc[1:idx, :]
merged_test_set = merged_data.iloc[idx:, :]

# ==================== 数据标准化 ====================
sc = MinMaxScaler(feature_range=(0, 1))
yuan_sc = MinMaxScaler(feature_range=(0, 1))
alpha_sc = MinMaxScaler(feature_range=(0, 1))
merged_sc = MinMaxScaler(feature_range=(0, 1))  # 新增merged数据的Scaler

# 标准化数据
training_set_scaled = sc.fit_transform(training_set)
testing_set_scaled = sc.transform(test_set) if len(test_set) > 0 else np.array([])

yuan_training_set_scaled = yuan_sc.fit_transform(yuan_training_set)
yuan_testing_set_scaled = yuan_sc.transform(yuan_test_set) if len(yuan_test_set) > 0 else np.array([])

filtered_alpha_features_scaled = alpha_sc.fit_transform(filtered_alpha_features_train)
filtered_alpha_features_test_scaled = alpha_sc.transform(filtered_alpha_features_test) if len(filtered_alpha_features_test) > 0 else np.array([])

merged_training_set_scaled = merged_sc.fit_transform(merged_training_set)
merged_testing_set_scaled = merged_sc.transform(merged_test_set) if len(merged_test_set) > 0 else np.array([])

# ==================== 数据准备 ====================
def safe_data_split(data, n_steps):
    if len(data) == 0:
        return np.array([]), np.array([])
    
    X, y = [], []
    max_possible = len(data) - n_steps
    if max_possible <= 0:
        print(f"警告: 数据不足，需要至少{n_steps+1}个样本，当前只有{len(data)}个")
        return np.array([]), np.array([])
    
    for i in range(max_possible):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps])
    return np.array(X), np.array(y)

print("\n=== 准备训练数据 ===")
X_train, y_train = safe_data_split(training_set_scaled, n_timestamp)
yuan_X_train, yuan_y_train = safe_data_split(yuan_training_set_scaled, n_timestamp)
filtered_X_train, filtered_y_train = safe_data_split(filtered_alpha_features_scaled, n_timestamp)
merged_X_train, merged_y_train = safe_data_split(merged_training_set_scaled, n_timestamp)  # 新增merged数据

print("\n=== 准备测试数据 ===")
X_test, y_test = safe_data_split(testing_set_scaled, n_timestamp) if testing_set_scaled.size > 0 else (np.array([]), np.array([]))
yuan_X_test, yuan_y_test = safe_data_split(yuan_testing_set_scaled, n_timestamp) if yuan_testing_set_scaled.size > 0 else (np.array([]), np.array([]))
filtered_X_test, filtered_y_test = safe_data_split(filtered_alpha_features_test_scaled, n_timestamp) if filtered_alpha_features_test_scaled.size > 0 else (np.array([]), np.array([]))
merged_X_test, merged_y_test = safe_data_split(merged_testing_set_scaled, n_timestamp) if merged_testing_set_scaled.size > 0 else (np.array([]), np.array([]))  # 新增merged数据

# ==================== 数据重塑 ====================
print("\n=== 重塑数据形状 ===")
# ARIMA残差数据
if X_train.size > 0:
    X_train = X_train.reshape(X_train.shape[0], n_timestamp, 1)
if X_test.size > 0:
    X_test = X_test.reshape(X_test.shape[0], n_timestamp, 1)

# 原始股票数据
if yuan_X_train.size > 0:
    yuan_X_train = yuan_X_train.reshape(yuan_X_train.shape[0], n_timestamp, 5)
if yuan_X_test.size > 0:
    yuan_X_test = yuan_X_test.reshape(yuan_X_test.shape[0], n_timestamp, 5)

# 特征工程数据
if filtered_X_train.size > 0:
    n_alpha_features = filtered_alpha_features_scaled.shape[1]
    try:
        filtered_X_train = filtered_X_train.reshape(filtered_X_train.shape[0], n_timestamp, n_alpha_features)
    except ValueError as e:
        print(f"重塑错误: {e}")
        actual_features = filtered_X_train.size // (filtered_X_train.shape[0] * n_timestamp)
        filtered_X_train = filtered_X_train.reshape(filtered_X_train.shape[0], n_timestamp, actual_features)

if filtered_X_test.size > 0:
    try:
        filtered_X_test = filtered_X_test.reshape(filtered_X_test.shape[0], n_timestamp, n_alpha_features)
    except ValueError as e:
        actual_features = filtered_X_test.size // (filtered_X_test.shape[0] * n_timestamp)
        filtered_X_test = filtered_X_test.reshape(filtered_X_test.shape[0], n_timestamp, actual_features)

# 新增merged数据
if merged_X_train.size > 0:
    merged_X_train = merged_X_train.reshape(merged_X_train.shape[0], n_timestamp, 5)
if merged_X_test.size > 0:
    merged_X_test = merged_X_test.reshape(merged_X_test.shape[0], n_timestamp, 5)

# ==================== 模型构建 ====================
print("\n=== 构建模型 ===")
# 1. ARIMA残差模型
arima_model = Sequential()
if model_type == 3:
    arima_model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(n_timestamp, 1)))
    arima_model.add(Dropout(0.2))
else:
    arima_model.add(LSTM(64, input_shape=(n_timestamp, 1)))
arima_model.add(Dense(1))
arima_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 2. 原始数据模型
yuan_model = Sequential()
yuan_model.add(LSTM(64, input_shape=(n_timestamp, 5)))
yuan_model.add(Dense(5))
yuan_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 3. 特征工程模型
alpha_model = Sequential()
if filtered_X_train.size > 0:
    alpha_model.add(LSTM(64, input_shape=(n_timestamp, filtered_X_train.shape[2])))
    alpha_model.add(Dense(filtered_X_train.shape[2]))
    alpha_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 4. 新增merged数据模型
merged_model = Sequential()
merged_model.add(LSTM(64, input_shape=(n_timestamp, 5)))
merged_model.add(Dense(5))
merged_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# 打印模型结构
print("\nARIMA残差模型结构:")
arima_model.summary()
print("\n原始数据模型结构:")
yuan_model.summary()
if filtered_X_train.size > 0:
    print("\n特征工程模型结构:")
    alpha_model.summary()
print("\nMerged数据模型结构:")
merged_model.summary()

# ==================== 模型训练 ====================
print("\n=== 训练模型 ===")
# 1. 训练ARIMA残差模型
if X_train.size > 0:
    print("\n训练ARIMA残差模型...")
    arima_history = arima_model.fit(
        X_train, y_train,
        epochs=n_epochs,
        batch_size=32,
        validation_data=(X_test, y_test) if X_test.size > 0 else None,
        verbose=1
    )

# 2. 训练原始数据模型
if yuan_X_train.size > 0:
    print("\n训练原始数据模型...")
    yuan_history = yuan_model.fit(
        yuan_X_train, yuan_y_train,
        epochs=n_epochs,
        batch_size=32,
        validation_data=(yuan_X_test, yuan_y_test) if yuan_X_test.size > 0 else None,
        verbose=1
    )

# 3. 训练特征工程模型
if filtered_X_train.size > 0:
    print("\n训练特征工程模型...")
    alpha_history = alpha_model.fit(
        filtered_X_train, filtered_y_train,
        epochs=n_epochs,
        batch_size=32,
        validation_data=(filtered_X_test, filtered_y_test) if filtered_X_test.size > 0 else None,
        verbose=1
    )

# 4. 训练merged数据模型
if merged_X_train.size > 0:
    print("\n训练Merged数据模型...")
    merged_history = merged_model.fit(
        merged_X_train, merged_y_train,
        epochs=n_epochs,
        batch_size=32,
        validation_data=(merged_X_test, merged_y_test) if merged_X_test.size > 0 else None,
        verbose=1
    )

# ==================== 模型评估 ====================
print("\n=== 模型评估 ===")
def evaluate_model(model, X_test, y_test, model_name):
    if X_test.size > 0:
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
        print(f"{model_name} - 测试集损失: {test_loss:.4f}, MAE: {test_mae:.4f}")
    else:
        print(f"{model_name} - 无测试数据")

evaluate_model(arima_model, X_test, y_test, "ARIMA残差模型")
evaluate_model(yuan_model, yuan_X_test, yuan_y_test, "原始数据模型")
if filtered_X_train.size > 0:
    evaluate_model(alpha_model, filtered_X_test, filtered_y_test, "特征工程模型")
evaluate_model(merged_model, merged_X_test, merged_y_test, "Merged数据模型")

# ==================== 结果可视化 ====================
print("\n=== 结果可视化 ===")
def plot_history_comparison(histories, labels):
    plt.figure(figsize=(12, 6))
    
    # 绘制训练损失
    plt.subplot(1, 2, 1)
    for history, label in zip(histories, labels):
        if history is not None:
            plt.plot(history.history['loss'], label=f'{label}训练集')
    plt.title('训练损失对比')
    plt.ylabel('损失')
    plt.xlabel('周期')
    plt.legend()
    
    # 绘制验证损失
    plt.subplot(1, 2, 2)
    for history, label in zip(histories, labels):
        if history is not None and 'val_loss' in history.history:
            plt.plot(history.history['val_loss'], label=f'{label}验证集')
    plt.title('验证损失对比')
    plt.ylabel('损失')
    plt.xlabel('周期')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 收集所有训练历史
histories = [
    arima_history if 'arima_history' in locals() else None,
    yuan_history if 'yuan_history' in locals() else None,
    alpha_history if 'alpha_history' in locals() else None,
    merged_history if 'merged_history' in locals() else None
]
labels = ['ARIMA残差', '原始数据', '特征工程', 'Merged数据']

# 绘制对比图
plot_history_comparison(histories, labels)

# 预测结果对比
def plot_predictions_comparison(models, X_tests, y_tests, model_names):
    plt.figure(figsize=(18, 4))
    
    for i, (model, X_test, y_test, name) in enumerate(zip(models, X_tests, y_tests, model_names)):
        if X_test.size == 0:
            continue
            
        plt.subplot(1, 4, i+1)
        predictions = model.predict(X_test)
        
        # 对于多输出模型，只取第一个特征
        if predictions.ndim > 2:
            predictions = predictions[:, 0]
            y_test = y_test[:, 0]
            
        plt.plot(y_test, label='真实值')
        plt.plot(predictions, label='预测值', alpha=0.7)
        plt.title(f'{name}预测对比')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# 收集所有模型和测试数据
models = [arima_model, yuan_model, alpha_model, merged_model]
X_tests = [X_test, yuan_X_test, filtered_X_test, merged_X_test]
y_tests = [y_test, yuan_y_test, filtered_y_test, merged_y_test]
model_names = ['ARIMA残差', '原始数据', '特征工程', 'Merged数据']

# 绘制预测对比图
plot_predictions_comparison(models, X_tests, y_tests, model_names)

# ==================== 保存结果 ====================
print("\n=== 保存预测结果 ===")
def save_predictions(model, X_test, scaler, data_index, filename):
    if X_test.size > 0:
        predictions = model.predict(X_test)
        if predictions.ndim > 2:
            predictions = predictions[:, 0]  # 取第一个特征
        
        # 逆标准化
        if scaler is not None:
            dummy = np.zeros((len(predictions), scaler.n_features_in_))
            dummy[:, 0] = predictions.flatten()  # 假设预测的是第一个特征
            predictions = scaler.inverse_transform(dummy)[:, 0]
        
        # 创建DataFrame
        pred_df = pd.DataFrame({
            'date': data_index[idx:idx+len(predictions)],
            'actual': y_test.flatten() if y_test.size > 0 else np.nan,
            'predicted': predictions.flatten()
        })
        pred_df.to_csv(filename, index=False)
        print(f"已保存: {filename}")

save_predictions(arima_model, X_test, sc, data.index, "arima_predictions.csv")
save_predictions(yuan_model, yuan_X_test, yuan_sc, yuan_data.index, "yuan_predictions.csv")
if filtered_X_test.size > 0:
    save_predictions(alpha_model, filtered_X_test, alpha_sc, filtered_alpha_features.index, "alpha_predictions.csv")
save_predictions(merged_model, merged_X_test, merged_sc, merged_data.index, "merged_predictions.csv")