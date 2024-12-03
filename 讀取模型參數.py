from tensorflow.keras.models import load_model  # type: ignore

# 指定.keras文件的路徑
model_path = 'final_best_model_fold_5.keras'

# 加載模型
model = load_model(model_path)

# 顯示模型的架構
model.summary()

# 簡化顯示模型的超參數
config = model.get_config()
print("超參數:")
for layer in config['layers']:
    layer_name = layer['class_name']
    layer_config = layer['config']
    units = layer_config.get('units', 'N/A')
    activation = layer_config.get('activation', 'N/A')
    dropout_rate = layer_config.get('rate', 'N/A') if layer_name == 'Dropout' else 'N/A'
    print(f"Layer: {layer_name}, Units: {units}, Activation: {activation}, Dropout Rate: {dropout_rate}\n")

# 輸出模型的學習率
learning_rate = model.optimizer.learning_rate.numpy()
print(f"模型的學習率: {learning_rate}")
