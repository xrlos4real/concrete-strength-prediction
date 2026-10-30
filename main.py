from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
import matplotlib.pyplot as plt
import pandas as pd
import joblib


dataset = fetch_ucirepo(id = 165)

X = dataset.data.features
y = dataset.data.targets.iloc[:, 0]

# Feature Engineering
X['Water_Binder_Ratio'] = (
    X['Water'] / 
    (
        X['Cement'] + X['Blast Furnace Slag'] + X['Fly Ash']
    )
)

# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 把输入特征和预测目标合并
# df = X.copy()
# df["Strength"] = y

# # 检查完整数据重复
# print("重复的输入特征:", X.duplicated().sum())
# print("完整重复的样本:", df.duplicated().sum())# kf = KFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )

# --------KFold-----------
# kf = KFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )
# ------------------------

# Linear Regression

# lg_model = LinearRegression()


# lg_scores = cross_val_score(
#         lg_model,
#         X_train_val,
#         y_train_val,
#         cv=kf,
#         scoring='neg_mean_absolute_error',
#         n_jobs=1
#     )

# lg_mae = -lg_scores.mean()

# print('MAE(Linear Regression):', lg_mae)

# Random Forest Regressor


rf_model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    min_samples_leaf=1,
    n_jobs=-1,
    max_depth=12
)

# rf_scores = cross_val_score(
#     rf_model,
#     X_train_val,
#     y_train_val,
#     cv=kf,
#     scoring='neg_mean_absolute_error',
#     n_jobs=1
# )
# print("新特征 CV MAE:", -rf_scores.mean())


# final model 
rf_model.fit(X_train_val, y_train_val)

y_pred = rf_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

print(f'MAE(Random Forest(Final_Model)):', mae)



# ---------绘图--------------
# 1. 绘制真实值和预测值的散点
plt.scatter(y_test, y_pred, alpha=0.6)

# 2. 绘制理想预测线
plt.plot([0, 90], [0, 90], 'r--')

# 3. 添加图表信息
plt.xlabel('Actual Strength(MPa)')
plt.ylabel('Predicted Stength(MPa)')
plt.title('Random Forest: Actual vs Predicted Strength')

plt.show()
# ----------------------------

# 找出误差最大的五个样本
# results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# results['Absolute Error'] = abs(results['Actual'] - results['Predicted'])
# print(
#     'Top 5 samples with the largest absolute errors:\n',
#     results.sort_values('Absolute Error', ascending=False).head(5)
# )


# 提取随机森林的特征重要性
importances = rf_model.feature_importances_

# 将特征名称与重要性对应起来
feature_importance = pd.Series(
    importances,
    index=X.columns
)
# 按重要性从小到大排列
feature_importance = feature_importance.sort_values()

print(feature_importance)

feature_importance.plot(kind='barh', figsize=(10, 6))

plt.title('Random Forest: Feature Importance')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.show()


# 保存模型
joblib.dump({
    'model': rf_model,
    'features': X_train_val.columns.tolist()
    }, 
    'RandomForestModel.joblib')