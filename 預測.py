import pandas as pd
import numpy as np
import tensorflow as tf
import tkinter as tk

from tkinter import filedialog
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# 設置支持中文的字體
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

root = tk.Tk()
root.withdraw()

# 顯示檔案選擇對話框，讓用戶選擇檔案
model_path = filedialog.askopenfilename(title="選擇檔案", filetypes=[("All Files", "*.*")])
model = tf.keras.models.load_model(model_path)

def Download_data():
    # 顯示檔案選擇對話框，讓用戶選擇檔案
    data_path  = filedialog.askopenfilename(title="選擇檔案", filetypes=[("All Files", "*.*")])

    # 顯示選擇的檔案路徑
    print(f"選擇的檔案路徑: {data_path }")
    
    return data_path 
def data_prepare(data : pd.DataFrame):
    data.drop([0,1] , inplace=True)
    dataset_X = data[[ 'Close' , 'Volume' ,'Open', 'Low' ,'High']]
    dataset_y = data[[ 'Close' ]]
   
    return dataset_X  , dataset_y
def create_sliding_windows(X, y, window_size):
    X_windows = []
    y_windows = []
    for i in range(window_size, len(X)):
        X_windows.append(X.iloc[i-window_size:i].values)  # X: 包含前 window_size 步的資料
        y_windows.append(y.iloc[i].values)                # y: 當前步的 Close 預測值
    return np.array(X_windows), np.array(y_windows)


def predict_next_n_days(model, scaler_X, scaler_y, dataset_X, n_days):
    last_90_days = dataset_X[-90:]  # 初始化最後90天的資料
    predictions = []
    
    for _ in range(n_days):
        # 將最後90天的數據縮放
        last_90_days_scaled = scaler_X.transform(last_90_days)
        # 預測下一天價格
        next_day_prediction = model.predict(np.array([last_90_days_scaled]))
        next_day_price = scaler_y.inverse_transform(next_day_prediction)[0][0]
        predictions.append(next_day_price)
        
        # 更新輸入，將預測值加入最後90天的資料
        next_day_data = last_90_days.iloc[1:].copy()  # 刪除最舊的一天
        next_day_data.loc[len(next_day_data)] = [next_day_price] + [0] * (last_90_days.shape[1] - 1)  # 添加新預測的價格
        last_90_days = next_day_data
    
    return predictions

def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    within_tolerance = np.abs((y_pred - y_true) / y_true) <= tolerance
    accuracy = np.mean(within_tolerance) * 100  # 轉為百分比
    return accuracy
# 主程式
df = pd.read_csv(Download_data())
dataset_X, dataset_y = data_prepare(df)
X, y = create_sliding_windows(dataset_X, dataset_y, 90)



scaler_X = MinMaxScaler(feature_range=(0, 1))
X = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
scaler_y = MinMaxScaler(feature_range=(0, 1))
y = scaler_y.fit_transform(y.reshape(-1, 1))

pre = model.predict(X)
predicted_prices = scaler_y.inverse_transform(pre)
accc = scaler_y.inverse_transform(y)

plt.figure(figsize=(15, 6))
plt.plot(accc, label='實際價格', color='blue')
plt.plot(predicted_prices, label='預測價格', color='orange')
plt.title("股票價格預測")
plt.xlabel('時間')
plt.ylabel('價格')
plt.legend()
plt.show()


accuracy = calculate_accuracy(accc , predicted_prices)

print(f'準確率 : {accuracy :2f}')

# 預測未來七天價格
seven_day_predictions = predict_next_n_days(model, scaler_X, scaler_y, dataset_X, 7)

print("未來七天預測價格:")
for i, price in enumerate(seven_day_predictions, 1):
    print(f"第{i}天: {price:.2f}")