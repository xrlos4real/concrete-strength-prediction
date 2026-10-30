import joblib
import pandas as pd
from pathlib import Path

# 1. 加载训练好的模型
model_path = Path(__file__).parent / 'RandomForestModel.joblib'

saved_data = joblib.load(model_path)

model = saved_data['model']
features_names = saved_data['features']

# 2. 接收用户输入的样本数据
sample = {}
features = features_names[:-1]  # 排除目标变量

for feature in features:
    value = float(input(f'Enter value for {feature}: '))
    sample[feature] = value

# 3. 计算水胶比
binder = (
    sample["Cement"]
    + sample["Blast Furnace Slag"]
    + sample["Fly Ash"]
)
if binder == 0:
    raise ValueError('Binder must be greater than zero')

sample["Water_Binder_Ratio"] = sample['Water'] / binder

# 4. 将样本转换为 DataFrame
X_new = pd.DataFrame([sample])
X_new = X_new[features_names]  # 确保特征顺序与训练时一致

# 5. 使用模型进行预测
prediction = model.predict(X_new)

print(f'Predicted Concrete Strength: {prediction[0]:.2f} MPa')