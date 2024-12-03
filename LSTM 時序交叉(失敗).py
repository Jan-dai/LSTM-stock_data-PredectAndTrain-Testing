import tensorflow as tf
import tkinter as tk
import pandas as pd
import numpy as np
import time, os
from tkinter import filedialog
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. 執行資料處理
class DataProcessor:
    def __init__(self, file_path=None):
        """
        初始化 DataProcessor 類別，載入並處理資料。

        Args:
            file_path (str, optional): 檔案路徑。如果未提供，則會顯示檔案選擇對話框讓使用者選擇。
                                      預設為 None。
        """
        if file_path is None:
            file_path = self._download_data()  # 如果沒有提供檔案路徑，則呼叫 _download_data 函數讓使用者選擇檔案
        self._load_and_prepare_data(file_path)  # 載入並處理選擇的檔案

    def _download_data(self):
        """
        顯示檔案選擇對話框，讓使用者選擇檔案。
        """
        root = tk.Tk()
        root.withdraw()  # 隱藏主視窗
        file_path = filedialog.askopenfilename(title="選擇檔案", filetypes=[("CSV Files", "*.csv")])  # 顯示檔案選擇對話框，讓使用者選擇 CSV 檔案

        if not file_path:
            print("請選擇檔案！")
            exit()  # 如果使用者沒有選擇檔案，則終止程式

        # 檢查檔案類型
        if not file_path.lower().endswith('.csv'):
            print("選擇檔案類型錯誤！請選擇 CSV 格式的檔案。")
            exit()  # 如果檔案類型不是 CSV，則終止程式

        print(f"選擇的檔案: {file_path}")
        return file_path

    def _load_and_prepare_data(self, file_path):
        """
        載入資料並進行預處理。
        """
        try:
            self.data = pd.read_csv(file_path)  # 使用 pandas 讀取 CSV 檔案
            self.dataset_X, self.dataset_y = self._prepare_data(self.data)  # 呼叫 _prepare_data 函數處理資料
        except Exception as e:
            print(f"載入檔案失敗: {e}")
            raise  # 發生錯誤時，拋出異常

    def _prepare_data(self, data: pd.DataFrame):
        """
        預處理資料，去除前兩行並提取特徵和目標變數。
        """
        data.drop([0, 1], inplace=True)  # 去除資料的前兩行
        dataset_X = data[['Close', 'Volume', 'Open', 'Low', 'High']]  # 提取特徵 (Close, Volume, Open, Low, High)
        dataset_y = data[['Close']]  # 提取目標變數 (Close)
        return dataset_X, dataset_y

    def create_sliding_windows(self, window_size):
        """
        創建滑動窗口數據。
        """
        X_windows = []  # 儲存特徵的滑動窗口
        y_windows = []  # 儲存目標變數的滑動窗口
        for i in range(window_size, len(self.dataset_X)):
            X_windows.append(self.dataset_X.iloc[i-window_size:i].values)  # X: 包含前 window_size 步的資料
            y_windows.append(self.dataset_y.iloc[i].values)                # y: 當前步的 Close 預測值
        return np.array(X_windows), np.array(y_windows)  # 將滑動窗口數據轉換為 NumPy 陣列


# 2. LSTM 模型
class LSTMModel:
    def __init__(self, look_back, early_stopping=None, model_checkpoint=None, folder_path=None, fold=1):
        """
        初始化 LSTMModel 類別，創建 LSTM 模型。

        Args:
            look_back (int): 滑動窗口大小。
            early_stopping (tf.keras.callbacks.EarlyStopping, optional): 早停機制。預設為 None。
            model_checkpoint (tf.keras.callbacks.ModelCheckpoint, optional): 模型檢查點。預設為 None。
            folder_path (str, optional): 模型儲存路徑。預設為 None。
            fold (int, optional): 折疊編號。預設為 1。
        """
        self.look_back = look_back  # 滑動窗口大小
        self.early_stopping = early_stopping  # 早停機制
        self.model_checkpoint = model_checkpoint  # 模型檢查點
        self.folder_path = folder_path  # 模型儲存路徑
        self.fold = fold  # 折疊編號
        self.log_df = pd.DataFrame()  # 訓練日誌
        self.model = self._create_model()  # 創建 LSTM 模型

    def _create_model(self):
        """
        創建並編譯 LSTM 模型。
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(self.look_back, 5)),  # 第一個 LSTM 層，輸出序列
            tf.keras.layers.BatchNormalization(),  # 批次標準化
            tf.keras.layers.Dropout(rate=0.4),  # 丟棄 40% 的神經元
            tf.keras.layers.LSTM(units=64, return_sequences=False),  # 第二個 LSTM 層，不輸出序列
            tf.keras.layers.BatchNormalization(),  # 批次標準化
            tf.keras.layers.Dropout(rate=0.2),  # 丟棄 20% 的神經元
            tf.keras.layers.Dense(units=32, activation='relu'),  # 全連接層，使用 ReLU 激活函數
            tf.keras.layers.Dense(units=1)  # 輸出層
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 使用 Adam 優化器
            loss=tf.keras.losses.Huber(delta=1.0),  # 使用 Huber 損失函數
            metrics=['mean_absolute_error', 'mean_squared_error']  # 使用平均絕對誤差和均方誤差作為評估指標
        )
        return model

    def model_training(self, X_train, y_train, validation_data, batch_size=32, epochs=30):
        """
        訓練模型並記錄日誌。
        """
        history = self.model.fit(
            x=X_train, y=y_train,
            batch_size=batch_size,  # 批次大小
            epochs=epochs,  # 訓練週期
            callbacks=[self.early_stopping, self.model_checkpoint],  # 使用早停機制和模型檢查點
            validation_data=validation_data,  # 驗證數據
            verbose=1  # 顯示訓練過程
        )
        self.log_df = pd.DataFrame({
            'epoch': range(1, len(history.history['loss']) + 1),
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        })  # 將訓練日誌儲存到 DataFrame 中

    def model_result(self, X_test, y_test, scaler_y):
        """
        在測試數據上評估模型並保存預測結果。
        """
        model_path = f'{self.folder_path}/fold_{self.fold}/best_model_fold_{self.fold}.keras'  # 模型儲存路徑
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型未找到: {model_path}")  # 如果模型不存在，則拋出異常
        model = tf.keras.models.load_model(model_path)  # 載入模型

        predicted_prices = scaler_y.inverse_transform(model.predict(X_test))  # 預測股票價格
        actual_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))  # 實際股票價格

        test_mse = mean_squared_error(actual_prices, predicted_prices)  # 計算均方誤差
        test_mae = mean_absolute_error(actual_prices, predicted_prices)  # 計算平均絕對誤差

        log_path = f'{self.folder_path}/fold_{self.fold}/training_log_fold_{self.fold}.csv'  # 訓練日誌儲存路徑
        self.log_df.to_csv(log_path, index=False)  # 將訓練日誌儲存到 CSV 檔案

        with open(f'{self.folder_path}/fold_{self.fold}/training_result_fold_{self.fold}.txt', mode="w", encoding="utf-8") as pen:
            pen.write(f"測試集的均方誤差 : {test_mse:.2f}\n測試集的平均絕對誤差 : {test_mae:.2f}")  # 將測試結果儲存到 TXT 檔案

        return test_mse, test_mae  # 返回均方誤差和平均絕對誤差


def create_data_folder():
    """
    創建資料夾以儲存訓練和測試數據。
    """
    base_folder = "C:\\Users\\user\\Downloads\\python_new-begain\\pro-3-datasave\\AI_TRAIN_TEST_SET00"  # 資料夾基本路徑
    i = 0
    while os.path.exists(f"{base_folder}{i:02d}"):  # 檢查資料夾是否存在
        i += 1
    new_folder = f"{base_folder}{i:02d}"  # 創建新的資料夾
    os.makedirs(new_folder)  # 創建資料夾
    return new_folder


def main():
    """
    主函式，執行資料處理、模型訓練和評估。
    """
    data_processor = DataProcessor()  # 初始化 DataProcessor 類別
    window_size = 90  # 滑動窗口大小
    X_windows, y_windows = data_processor.create_sliding_windows(window_size)  # 創建滑動窗口數據
    folder_path = create_data_folder()  # 創建資料夾
    fold = 1  # 折疊編號
    for train_index, test_index in TimeSeriesSplit(n_splits=4).split(X_windows):  # 使用 TimeSeriesSplit 將數據劃分為訓練集和測試集
        X_train, X_test = X_windows[train_index], X_windows[test_index]  # 提取訓練集和測試集的特徵
        y_train, y_test = y_windows[train_index], y_windows[test_index]  # 提取訓練集和測試集的目標變數
        val_size = int(0.2 * len(X_train))  # 驗證集大小
        X_val, y_val = X_train[-val_size:], y_train[-val_size:]  # 提取驗證集
        X_train, y_train = X_train[:-val_size], y_train[:-val_size]  # 移除驗證集

        scaler_X = MinMaxScaler(feature_range=(0, 1))  # 初始化 MinMaxScaler，將特徵縮放到 0 到 1 之間
        X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)  # 縮放訓練集特徵
        X_val = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)  # 縮放驗證集特徵
        X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)  # 縮放測試集特徵

        scaler_y = MinMaxScaler(feature_range=(0, 1))  # 初始化 MinMaxScaler，將目標變數縮放到 0 到 1 之間
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))  # 縮放訓練集目標變數
        y_val = scaler_y.transform(y_val.reshape(-1, 1))  # 縮放驗證集目標變數
        y_test = scaler_y.transform(y_test.reshape(-1, 1))  # 縮放測試集目標變數

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=40, restore_best_weights=True
        )  # 初始化 EarlyStopping，監控驗證集損失，如果損失在 40 個週期內沒有改善，則停止訓練
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f'{folder_path}/fold_{fold}/best_model_fold_{fold}.keras',
            monitor='val_loss',
            save_best_only=True
        )  # 初始化 ModelCheckpoint，保存具有最佳驗證集損失的模型
        os.makedirs(f"{folder_path}/fold_{fold}", exist_ok=True)  # 創建資料夾

        lstm_model = LSTMModel(
            look_back=window_size,
            early_stopping=early_stopping,
            model_checkpoint=model_checkpoint,
            folder_path=folder_path,
            fold=fold
        )  # 初始化 LSTMModel 類別

        lstm_model.model_training(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=200)  # 訓練模型
        test_mse, test_mae = lstm_model.model_result(X_test, y_test, scaler_y)  # 評估模型
        print(f"測試集的均方誤差 : {test_mse:.2f}\n測試集的平均絕對誤差 : {test_mae:.2f}")  # 輸出測試結果
        fold += 1  # 更新折疊編號


if __name__ == "__main__":
    start_time = time.time()  # 記錄開始時間
    main()  # 執行主函數
    end_time = time.time()  # 記錄結束時間
    formatted_time = str(time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)))  # 格式化時間
    print(f"總訓練時間: {formatted_time}")  # 輸出總訓練時間
