import time , os , keras 

import matplotlib.pyplot as plt
import keras_tuner as kt
import tensorflow as tf
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error , make_scorer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from datetime import datetime, timedelta



# 設置支持中文的字體
plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

class CustomObjectiveCallback(keras.callbacks.Callback):
    def __init__(self, tuner, w1, w2):
        super().__init__()
        self.tuner = tuner
        self.w1 = w1
        self.w2 = w2
        self.best_score = -np.inf
        self.best_hps = None

    def on_epoch_end(self, epoch, logs=None):
        # 計算自定義的目標分數
        score = self.w1 * logs['val_accuracy'] - self.w2 * logs['val_loss']
        
        # 如果新的分數更高，則更新最佳超參數組合
        if score > self.best_score:
            self.best_score = score
            self.best_hps = self.tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters


# 1. 數據獲取和預處理
def get_stock_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    def replace_zeros_with_median_pandas(df: pd.DataFrame, col: str) -> pd.DataFrame:
        def replace_zero_with_median(index, value) -> int:
            if value == 0:
                return int(np.median([df.at[index - timedelta(1), col], df.at[index + timedelta(1), col]]))
            return int(value)

        df = df.apply(lambda row: replace_zero_with_median(row.name, row[col]), axis=1)
        return df
    
    ddf = df.loc[start_date.strftime('%Y-%m-%d %H:%M:%S')::,'Volume']
    ddf = replace_zeros_with_median_pandas(ddf , '2330.TW')
    
    df.loc[start_date.strftime('%Y-%m-%d %H:%M:%S')::,'Volume'] = ddf.values
    
    return df

def prepare_data(data, look_back=60):
    dataset = data[['Close', 'Volume']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i])
        y.append(dataset[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler

# 2. 超參數調整：創建可調整的 LSTM 模型
def create_model(hp, look_back):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(
            hp.Int('units_1', min_value=50, max_value=200, step=50), 
            return_sequences=True, 
            input_shape=(look_back, 2)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(hp.Float('dropout_1', min_value=0.3, max_value=0.7, step=0.1)),
        tf.keras.layers.LSTM(
            hp.Int('units_2', min_value=50, max_value=150, step=50), 
            return_sequences=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(hp.Float('dropout_2', min_value=0.3, max_value=0.7, step=0.1)),
        tf.keras.layers.Dense(15, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
        loss='mean_squared_error',
        metrics=['mean_absolute_error', 'mean_squared_error']) # add accuracy計算
    
    return model

# 3. 使用 Keras Tuner 進行超參數調整
def tune_hyperparameters(X, y, look_back):
    tuner = kt.Hyperband(
        lambda hp: create_model(hp, look_back),
        objective=kt.Objective('val_accuracy', direction='max'), #'val_loss'
        max_epochs=10,
        factor=3,
        directory='my_dir_3',
        project_name='stock_prediction'
    )
    
    tuner.search(X, y, epochs=50, batch_size=64, validation_split=0.2)
    
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters: ", best_hp.values)
    return best_hp

# 4. 預測視覺化
def plot_predictions(actual, predicted, title, filename):
    plt.figure(figsize=(15, 6))
    plt.plot(actual, label='實際價格', color='blue')
    plt.plot(predicted, label='預測價格', color='orange')
    plt.title(title)
    plt.xlabel('時間')
    plt.ylabel('價格')
    plt.legend()
    plt.savefig(filename)
    plt.close()

# 5. 損失函數視覺化  
def plot_loss_curve(train_loss, val_loss, title, filename):
    plt.figure(figsize=(15, 6))
    plt.plot(train_loss, label='訓練損失')
    plt.plot(val_loss, label='驗證損失')
    plt.title(title)
    plt.xlabel('訓練輪次')
    plt.ylabel('損失')
    plt.legend()
    plt.savefig(filename)
    plt.close()

# 檢查並創建資料夾
def create_data_folder():
    base_folder = "AI_TRAIN_TEST_SET00"
    i = 0
    while os.path.exists(f"{base_folder}{i:02d}"):
        i += 1
    new_folder = f"{base_folder}{i:02d}"
    os.makedirs(new_folder)
    return new_folder

# 主程序
def main():
    start_time = time.time() 
    
    symbol = '2330.TW'
    look_back = 90
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3000)
    
    print("正在下載股票數據...")
    df = get_stock_data(symbol, start_date, end_date)
    
    print("正在準備數據...")
    X, y, scaler = prepare_data(df, look_back)

    # 創建資料夾
    folder_path = create_data_folder()
    print(f"儲存資料將於資料夾: {folder_path}")
    
    # 進行 KFold 5 摺交叉驗證
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold = 1
    all_fold_results = []  # 儲存每一摺的結果
    all_hyperparameters = []  # 儲存每一摺的超參數
    
    for train_index, val_index in kf.split(X):
        print(f"\n正在進行第 {fold} 摺訓練...")
        
        # 使用 KFold 切分訓練集和驗證集
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # 再次將訓練集切分出測試集（70% 訓練集，15% 驗證集，15% 測試集）
        train_size = int(0.7 * len(X_train))
        X_train_split, X_test_split = X_train[:train_size], X_train[train_size:]
        y_train_split, y_test_split = y_train[:train_size], y_train[train_size:]

        # 調整超參數
        best_hp = tune_hyperparameters(X_train_split, y_train_split, look_back)
        
        # 創建最佳超參數的模型
        model = create_model(best_hp, look_back)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{folder_path}/fold_{fold}/best_model_fold_{fold}.keras', monitor='val_loss', save_best_only=True)

        # 創建摺數資料夾
        os.makedirs(f"{folder_path}/fold_{fold}", exist_ok=True)

        print("正在訓練模型...")
        history = model.fit(X_train_split, y_train_split, 
                            epochs=150, 
                            batch_size=64, 
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping, model_checkpoint],
                            verbose=1)
        
        log_data = {
            'epoch': range(1, len(history.history['loss']) + 1),
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        
        log_df = pd.DataFrame(log_data)
        log_df.to_csv(f'{folder_path}/fold_{fold}/training_log_fold_{fold}.csv', index=False)
        
        # 測試並記錄結果
        model = tf.keras.models.load_model(f'{folder_path}/fold_{fold}/best_model_fold_{fold}.keras')
        
        predicted_prices = model.predict(X_test_split)
        predicted_prices = scaler.inverse_transform(np.concatenate([predicted_prices, np.zeros_like(predicted_prices)], axis=1))[:, 0]
        actual_prices = scaler.inverse_transform(np.concatenate([y_test_split.reshape(-1, 1), np.zeros_like(y_test_split).reshape(-1, 1)], axis=1))[:, 0]
        
        test_mse = mean_squared_error(actual_prices, predicted_prices)
        test_mae = mean_absolute_error(actual_prices, predicted_prices)
        
        plot_predictions(actual_prices, predicted_prices, f'第 {fold} 摺 股票實際價格與預測價格對比', f'{folder_path}/fold_{fold}/price_comparison_fold_{fold}.png')
        plot_loss_curve(history.history['loss'], history.history['val_loss'], f'第 {fold} 摺 模型損失', f'{folder_path}/fold_{fold}/loss_curve_fold_{fold}.png')
        
        # 儲存最佳模型
        model.save(f'{folder_path}/fold_{fold}/final_best_model_fold_{fold}.keras')

        # 記錄每摺的評估數據
        fold_results = {
            'fold': fold,
            'train_loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1],
            'test_mse': test_mse,
            'test_mae': test_mae
        }
        all_fold_results.append(fold_results)
        
        fold += 1
    
    # 保存所有摺的結果
    fold_results_df = pd.DataFrame(all_fold_results)
    fold_results_df.to_csv(f'{folder_path}/fold_results.csv', index=False)
    
    # # 儲存超參數
    hyperparameters = best_hp.values
    all_hyperparameters.append(hyperparameters)
    hyperparameters_df = pd.DataFrame(all_hyperparameters)
    hyperparameters_df.to_csv(f'{folder_path}/hyperparameters.csv', index=False)
    
    end_time = time.time()
    total_time = int(end_time - start_time)
    formatted_time = str(timedelta(seconds=total_time))
    print(f"\n總訓練時間: {formatted_time}")

if __name__ == "__main__":
    main()
