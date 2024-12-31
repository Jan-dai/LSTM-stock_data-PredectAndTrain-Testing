import tensorflow as tf
import tkinter as tk
import pandas as pd
import numpy as np
import time, os
from tkinter import filedialog
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # 替換為 train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. 執行資料處理
class DataProcessor:
    def __init__(self, file_path=None):
        """
        初始化 DataProcessor 類別，載入並處理資料。
        """
        if file_path is None:
            file_path = self._download_data()
        self._load_and_prepare_data(file_path)

    def _download_data(self):
        """
        顯示檔案選擇對話框，讓使用者選擇檔案。
        """
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(title="選擇檔案", filetypes=[("CSV Files", "*.csv")])

        if not file_path:
            print("請選擇檔案！")
            exit()  # 終止程式

        # 檢查檔案類型
        if not file_path.lower().endswith('.csv'):
            print("選擇檔案類型錯誤！請選擇 CSV 格式的檔案。")
            exit()  # 終止程式

        print(f"選擇的檔案: {file_path}")
        return file_path

    def _load_and_prepare_data(self, file_path):
        """
        載入資料並進行預處理。
        """
        try:
            self.data = pd.read_csv(file_path)
            self.dataset_X, self.dataset_y = self._prepare_data(self.data)
        except Exception as e:
            print(f"載入檔案失敗: {e}")
            raise

    def _prepare_data(self, data: pd.DataFrame):
        """
        預處理資料，去除前兩行並提取特徵和目標變數。
        """
        data.drop([0, 1], inplace=True)
        dataset_X = data[['Close', 'Volume', 'Open', 'Low', 'High']]
        dataset_y = data[['Close']]
        return dataset_X, dataset_y

    def create_sliding_windows(self, window_size):
        """
        創建滑動窗口數據。
        """
        X_windows = []
        y_windows = []
        for i in range(window_size, len(self.dataset_X)):
            X_windows.append(self.dataset_X.iloc[i-window_size:i].values)  # X: 包含前 window_size 步的資料
            y_windows.append(self.dataset_y.iloc[i].values)                # y: 當前步的 Close 預測值
        return np.array(X_windows), np.array(y_windows)

# 2. LSTM 模型
class LSTMModel:
    def __init__(self, look_back, early_stopping=None, model_checkpoint=None, folder_path=None, fold=1):
        self.look_back = look_back
        self.early_stopping = early_stopping
        self.model_checkpoint = model_checkpoint
        self.folder_path = folder_path
        self.fold = fold
        self.log_df = pd.DataFrame()
        self.model = self._create_model()

    def _create_model(self):
        """
        創建並編譯 LSTM 模型。
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(self.look_back, 5)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.4),
            tf.keras.layers.LSTM(units=64, return_sequences=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Dense(units=32, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=tf.keras.losses.Huber(delta=1.0),
            metrics=['mean_absolute_error', 'mean_squared_error']
        )
        return model

    def model_training(self, X_train, y_train, validation_data, batch_size=32, epochs=30):
        """
        訓練模型並記錄日誌。
        """
        history = self.model.fit(
            x=X_train, y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[self.early_stopping, self.model_checkpoint],
            validation_data=validation_data,
            verbose=1
        )
        self.log_df = pd.DataFrame({
            'epoch': range(1, len(history.history['loss']) + 1),
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        })

    def model_result(self, X_test, y_test, scaler_y):
        """
        在測試數據上評估模型並保存預測結果。
        """
        model_path = f'{self.folder_path}/fold_{self.fold}/best_model_fold_{self.fold}.keras'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型未找到: {model_path}")
        model = tf.keras.models.load_model(model_path)

        predicted_prices = scaler_y.inverse_transform(model.predict(X_test))
        actual_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        test_mse = mean_squared_error(actual_prices, predicted_prices)
        test_mae = mean_absolute_error(actual_prices, predicted_prices)

        log_path = f'{self.folder_path}/fold_{self.fold}/training_log_fold_{self.fold}.csv'
        self.log_df.to_csv(log_path, index=False)

        with open(f'{self.folder_path}/fold_{self.fold}/training_result_fold_{self.fold}.txt', mode="w", encoding="utf-8") as pen:
            pen.write(f"測試集的均方誤差 : {test_mse:.2f}\n測試集的平均絕對誤差 : {test_mae:.2f}")

        return test_mse, test_mae

def create_data_folder():
    """
    創建資料夾以儲存訓練和測試數據。
    """
    base_folder = "C:\\Users\\user\\Downloads\\python_new-begain\\pro-3-datasave\\AI_TRAIN_TEST_SET00"
    i = 0
    while os.path.exists(f"{base_folder}{i:02d}"):
        i += 1
    new_folder = f"{base_folder}{i:02d}"
    os.makedirs(new_folder)
    return new_folder

def main():
    """
    主函式，執行資料處理、模型訓練和評估。
    """
    data_processor = DataProcessor()
    window_size = 90
    X_windows, y_windows = data_processor.create_sliding_windows(window_size)
    folder_path = create_data_folder()

    X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.2, random_state=42) # 使用 train_test_split 分割資料
    val_size = int(0.2 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=80, restore_best_weights=True
    )
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'{folder_path}/fold_1/best_model_fold_1.keras', # 移除 fold 迴圈，只儲存一個模型
        monitor='val_loss',
        save_best_only=True
    )
    os.makedirs(f"{folder_path}/fold_1", exist_ok=True)

    lstm_model = LSTMModel(
        look_back=window_size,
        early_stopping=early_stopping,
        model_checkpoint=model_checkpoint,
        folder_path=folder_path,
        fold=1 # 移除 fold 迴圈，fold 設為 1
    )

    lstm_model.model_training(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=500)
    test_mse, test_mae = lstm_model.model_result(X_test, y_test, scaler_y)
    print(f"測試集的均方誤差 : {test_mse:.2f}\n測試集的平均絕對誤差 : {test_mae:.2f}")
    # fold += 1

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    formatted_time = str(time.strftime('%H:%M:%S', time.gmtime(end_time - start_time)))
    print(f"總訓練時間: {formatted_time}")
