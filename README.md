# 1.

股票預測程式-測試實驗研究作

------------------------------------------------------------------
# 2. :

訓練測試，加入交易量為第2個特徵(第1特徵為收盤價)。

使用 Keras Tuner 進行超參數調整，並輸出最佳超參數調整數值；並使用K-Fold交叉驗證法用以模型評估。

K-Fold交叉驗證法:
一種通用的交叉驗證技術，用於評估模型的性能。將數據集分為 K 個相等的部分（folds），並多次訓練模型，每次使用某一個部分作為測試集，其餘部分作為訓練集。

EX:

        import numpy as np
        from sklearn.model_selection import KFold
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        # 假設數據
        X = np.arange(20).reshape(10, 2)  # 特徵數據
        y = np.arange(10)                # 標籤數據
        
        # 設置 KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        model = LinearRegression()
        fold_mse = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            # 劃分訓練集和測試集
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 訓練模型
            model.fit(X_train, y_train)
            
            # 評估性能
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            fold_mse.append(mse)
            
            print(f"Fold {fold + 1}, MSE: {mse:.4f}")
      
            print(f"平均MSE: {np.mean(fold_mse):.4f}")
    
優點:
有效利用數據：每個數據樣本既參與訓練也參與測試。
穩定性高：性能指標的平均值能更可靠地反映模型性能。

缺點:
訓練次數多：需要重複訓練 K 次，計算量大。不適合時序數據：會打亂數據的時間順序（對於時序數據需使用時序 K-Fold）。

結論:
由於資料形式與驗證法運行方式具有衝突的請況，因此改使用時序 K-Fold驗證法進行評估，並且只使用2個特徵資訊量不足因此改全使用(收盤價,交易量,開盤價, 最低價, 最高價)。
-------------------------------------------------------------------
# 3. :

訓練測試，特徵使用改為全使用(收盤價,交易量,開盤價, 最低價, 最高價)。
驗證法改使用時序 K-Fold驗證法進行評估。
取消使用Keras Tuner進行超參數調整，並輸出最佳超參數調整數值，改為直接設定超參數。

時序 K-Fold驗證法
EX:
        import numpy as np
        from sklearn.model_selection import TimeSeriesSplit
        # 假設有100筆時間序列資料
        data = np.arange(100)
        
        # 使用 TimeSeriesSplit
        n_splits = 5
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        for fold, (train_index, test_index) in enumerate(tscv.split(data)):
            print(f"Fold {fold+1}")
            print("Train indices:", train_index)
            print("Test indices:", test_index)

優點:
適合時序資料，能模擬實際預測場景。
減少資料洩漏風險。
缺點:
測試資料的樣本數量可能有限。
訓練集逐漸增大，可能影響計算效率。

結論:
---------------------------------------------------------------------------------
