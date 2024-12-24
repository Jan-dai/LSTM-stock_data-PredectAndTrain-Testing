import time, os
import matplotlib.pyplot as plt
import keras_tuner as kt
import tensorflow as tf

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
#from tensorflow.keras.regularizers import L1L2 , L2 # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error
#from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

from datetime import datetime, timedelta

# 設置支持中文的字體
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 0. 自訂計算準確率
def calculate_accuracy(y_true, y_pred, tolerance=0.05):
    within_tolerance = np.abs((y_pred - y_true) / y_true) <= tolerance
    accuracy = np.mean(within_tolerance) * 100  # 轉為百分比
    return accuracy

# 1. 數據獲取和預處理
def get_stock_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
  
    def replace_zeros_with_median(data, col):
        imputer = SimpleImputer(missing_values=0, strategy='median')
        imputer.fit(data[[col]])
        data[col] = imputer.transform(data[[col]]).astype(int)
    
    replace_zeros_with_median(df, "Volume")
 
    return df
def create_sliding_windows(data, look_back=60):
    dataset = data[['Close', 'Volume']].values
    X, y = [], []
    for i in range(look_back, len(dataset)):
        X.append(dataset[i - look_back:i])
        y.append(dataset[i, 0])
    
    X, y = np.array(X), np.array(y)
    return X, y
def K_Fold_Manual_TimeSeries(X, y, train_index, val_index, train_ratio=0.7, val_ratio=0.15):
    """
    修正後，確保每次分割符合 70% 訓練集、15% 驗證集、15% 測試集比例。
    """
    total_len = len(train_index) + len(val_index)

    # 計算每個集的大小
    train_size = int(total_len * train_ratio)
    val_size = int(total_len * val_ratio)
    test_size = total_len - train_size - val_size

    # 確定分割索引
    train_start = 0
    train_end = train_start + train_size
    val_end = train_end + val_size

    X_train = X[train_start:train_end]
    y_train = y[train_start:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:val_end + test_size]
    y_test = y[val_end:val_end + test_size]

    # 標準化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))

    # 儲存分割結果
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, scaler_y

# 2. 超參數調整：創建可調整的 LSTM 模型
def create_model(hp, look_back):
    model = tf.keras.Sequential()
    
    model.add(
        tf.keras.layers.LSTM(
            hp.Int('units_1', min_value=128, max_value=320, step=32),
            return_sequences=True,
            input_shape=(look_back, 2)
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(
        hp.Float('dropout_1', min_value=0.1, max_value=0.8, step=0.1)
        )
    )
    model.add(
        tf.keras.layers.LSTM(
            hp.Int('units_2', min_value=64, max_value=256, step=32),
            return_sequences=False
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(
        hp.Float('dropout_2', min_value=0.1, max_value=0.8, step=0.1)
        )
    )
    model.add(tf.keras.layers.Dense(
        hp.Int('dense_units', min_value=16, max_value=64, step=16),
        activation='relu'
        )
    )
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
            ),
        loss=tf.keras.losses.mean_squared_error,
        metrics=['mean_absolute_error', 'mean_squared_error']
    )
   
    return model

# 3. 使用 Keras Tuner 進行超參數調整
def tune_hyperparameters(X, y, look_back , folder_path):
    
    # tuner = kt.Hyperband(
    #     lambda hp: create_model(hp, look_back),  # 模型創建函數
    #     objective='val_loss', # # 調整目標 
    #     max_epochs=100, #100
    #     directory=os.path.join(folder_path, 'my_dir'), 
    #     project_name='stock_prediction',         
    #     overwrite=False                          # 是否覆蓋舊結果
    # )
    # tuner_1 = kt.RandomSearch(
    #     lambda hp: create_model(hp, look_back),  # 模型創建函數
    #     objective='val_loss', # # 調整目標 
    #     max_epochs=100, #100
    #     directory=os.path.join(folder_path, 'my_dir'), 
    #     project_name='stock_prediction',         
    #     overwrite=False    
    # )
    tuner_2 = kt.BayesianOptimization(
        hypermodel= lambda hp : create_model(hp, look_back),
        max_trials=2,
        objective='val_loss',
        max_retries_per_trial=1,
        directory=os.path.join(folder_path, 'my_dir'), 
        project_name='stock_prediction',         
        overwrite=False
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_val = scaler_y.transform(y_val.reshape(-1, 1))

    # tuner.search(
    #     X_train, y_train, 
    #     epochs=10, #40
    #     batch_size=64, #32
    #     validation_data=(X_val, y_val) , 
    #     callbacks = [early_stopping]
    # )
    tuner_2.search(
        X_train, y_train, 
        epochs=4, #40
        batch_size=64, #32
        validation_data=(X_val, y_val) , 
        callbacks = [early_stopping]
    )
   
    best_hp = tuner_2.get_best_hyperparameters(num_trials=1)[0]
    
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
# def create_data_folder():
    
    
#     base_folder = "C:\\Users\\user\\Downloads\\python_new-begain\\pro-3-datasave\\AI_TRAIN_TEST_SET00"
#     i = 0
#     while os.path.exists(f"{base_folder}{i:02d}"):
#         i += 1
#     new_folder = f"{base_folder}{i:02d}"
#     os.makedirs(new_folder)
#     return new_folder

def result_out(folder_path , all_fold_results, summary_stats):

    with open(f"{folder_path}\\results.txt", "w", encoding="utf-8") as file:
        for result in all_fold_results:
            file.write(f"\n第 {result['fold']} 摺結果:\n")
            file.write(f"測試集 MSE: {result['test_mse']:.4f}\n")
            file.write(f"測試集 MAE: {result['test_mae']:.4f}\n")
            file.write(f"準確率: {result['accuracy']:.2f}%\n")
            file.write(f"未來 7 天預測價格: {result['future_7_day_predictions']}\n\n")
        
        file.write("\n總結結果:\n")    
        
        for key, value in summary_stats.items():
            print(f"     {key}: {value:.4f}\n")
            file.write(f"     {key}: {value:.4f}\n")
         
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
    X, y = create_sliding_windows(df, look_back)

    # 創建資料夾
    #folder_path = create_data_folder()
    folder_path ="C:\\Users\\user\\Downloads\\python_new-begain\\pro-3-datasave\\AI_TRAIN_TEST_SET0000"
    print(f"儲存資料將於資料夾: {folder_path}")
    

    best_hp = tune_hyperparameters(
        X ,  y, 
        look_back, 
        folder_path
    )

    # 進行 時序 K-Fold 5 摺 交叉驗證
    #tscv = TimeSeriesSplit(n_splits=5)  
    # 進行 K-Fold 5 摺 交叉驗證
    k_fold = KFold(n_splits=5 ,shuffle=False)  
    
    fold = 1
    all_fold_results = []
    
    for train_index, val_index in k_fold.split(X):
    
        print(f"\n正在進行第 {fold} 摺訓練...")
        
        X_train , y_train , X_val , y_val , X_test , y_test , scaler_X , scaler_y = K_Fold_Manual_TimeSeries(X , y , train_index ,val_index )
#  驗證資料分割
        total = len(X_train) + len(X_val) + len(X_test)
        print(f"Fold {fold }:")
        print("X( 特徵集 )\n")
        print(f"Train: {len(X_train)} ({len(X_train) / total:.2%}), "
              f"Val: {len(X_val)} ({len(X_val) / total:.2%}), "
              f"Test: {len(X_test)} ({len(X_test) / total:.2%})")
        print("y( 標籤集 )\n")
        print(f"Train: {len(y_train)} ({len(y_train) / total:.2%}), "
              f"Val: {len(y_val)} ({len(y_val) / total:.2%}), "
              f"Test: {len(y_test)} ({len(y_test) / total:.2%})")
#  驗證資料分割
        model = create_model(best_hp, look_back)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            f'{folder_path}/fold_{fold}/best_model_fold_{fold}.h5',# .keras tensorflow 2.18
            monitor='val_loss', 
            save_best_only=True
        )
        os.makedirs(f"{folder_path}/fold_{fold}", exist_ok=True)

        print("正在訓練模型...")
        history = model.fit(X_train, y_train, 
                            epochs=8, #80
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
        
        model = tf.keras.models.load_model(f'{folder_path}/fold_{fold}/best_model_fold_{fold}.h5') #.keras tensorflow 2.18
        
        predicted_prices = model.predict(X_test)
        predicted_prices = scaler_y.inverse_transform(predicted_prices)
        actual_prices = scaler_y.inverse_transform(y_test.reshape(-1, 1))

        test_mse = mean_squared_error(actual_prices, predicted_prices)
        test_mae = mean_absolute_error(actual_prices, predicted_prices)
        
        accuracy = calculate_accuracy(actual_prices, predicted_prices)
        
        future_7_day_predictions = []
        last_known_sequence = X_test[-1]
        for _ in range(7):
            next_prediction = model.predict(last_known_sequence[np.newaxis, ...])[0, 0]
            future_7_day_predictions.append(int(scaler_y.inverse_transform([[next_prediction]])[0, 0]))
            last_known_sequence = np.roll(last_known_sequence, -1, axis=0)
            last_known_sequence[-1, 0] = next_prediction

        plot_predictions(
            actual_prices, predicted_prices, 
            title=f"第 {fold} 摺驗證集 - 實際與預測價格對比", 
            filename=f"{folder_path}/fold_{fold}/validation_vs_predicted_prices.png"
        )
        
        plot_loss_curve(
            history.history['loss'], history.history['val_loss'], 
            title=f"第 {fold} 摺訓練與驗證損失", 
            filename=f"{folder_path}/fold_{fold}/training_validation_loss.png"
        )
        
        result = {
            'fold': fold,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'accuracy': accuracy,
            'future_7_day_predictions': future_7_day_predictions
        }
        all_fold_results.append(result)
        fold += 1
    
    end_time = time.time()
    total_time = int(end_time - start_time)
    formatted_time = str(timedelta(seconds=total_time))
    print(f"\n總訓練時間: {formatted_time}")
    
    hyperparameters = best_hp.values
    hyperparameters_df = pd.DataFrame([hyperparameters])
    hyperparameters_df.to_csv(f'{folder_path}/best_hyperparameters.csv', index=False)
    
    
    summary_results = pd.DataFrame(all_fold_results)
    summary_stats = {
        '平均 MSE': summary_results['test_mse'].mean(),
        '平均 MAE': summary_results['test_mae'].mean(),
        '平均準確率': summary_results['accuracy'].mean()
    }
    
    result_out(  folder_path , all_fold_results, summary_stats)
    for result in all_fold_results:
        print(f"\n第 {result['fold']} 摺結果:")
        print(f"測試集 MSE: {result['test_mse']:.4f}")
        print(f"測試集 MAE: {result['test_mae']:.4f}")
        print(f"準確率: {result['accuracy']:.2f}%")
        print(f"未來 7 天預測價格: {result['future_7_day_predictions']}")

    print("\n總結結果:")

    for key, value in summary_stats.items():
        print(f"     {key}: {value:.4f}\n")

        

   
        

if __name__ == "__main__":
    main()
