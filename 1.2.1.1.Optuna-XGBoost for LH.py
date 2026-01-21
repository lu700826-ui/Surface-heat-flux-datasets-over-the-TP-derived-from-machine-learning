import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
import optuna
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import shap
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.feature_selection import RFE

df = pd.read_excel('F:/LHSH/0004.LSTM/jhy/LH_site_temp_smooth_0123.xlsx',engine='openpyxl', index_col=0) #, index_col=0)
df = df.iloc[:, 2:]
X=df
X = X.drop('OBS_lh_QC', axis=1)
X = X.drop('OBS_lh', axis=1)
X = X.iloc[:, 0:-1]# 所有列，从第4列开始
y= df.iloc[:,-1] # 第一列为目标变量

# 数据准备
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


# 定义 Optuna 优化目标函数，用 R² 作为目标
def objective(trial):
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "booster": "gbtree",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "n_estimators": trial.suggest_int("n_estimators", 50, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
    }

    model = xgb.XGBRegressor(**param)
    # 使用 R² 分数进行 10 折交叉验证
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring="r2")
    print(scores)
    return scores.mean()  # R² 越大越好


# 使用 Optuna 进行参数优化
study = optuna.create_study(direction="maximize")  # maximize R²
study.optimize(objective, n_trials=20)

# 打印最优参数
print("Best trial:")
print(study.best_trial.params)

# 用最优参数训练模型
best_params = study.best_trial.params
model = xgb.XGBRegressor(**best_params)

# 使用 RFE 进行特征选择
selector = RFE(model, n_features_to_select=12)  # 选择10个最重要的特征
selector = selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.support_]

# 筛选后的特征
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
# 将 X_test_selected 转换为 DataFrame，并赋予新的列名（筛选后的特征名）
X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
# 将 X_test_selected 保存为新的 Excel 文件
X_test_selected_df.to_excel('F:/LHSH/0004.LSTM/jhy/test0515/OPTUNA_XGBoost_LH_X_test_selector_0519.xlsx', index=False)


# 设置 evals 参数以便记录学习过程中的误差
evals = [(X_train_selected, y_train), (X_test_selected, y_test)]  # 训练集和测试集
evals_result = {}  # 存储每个迭代的训练和验证误差

# 初始化空的R²值列表
train_r2_scores = []
val_r2_scores = []

# 训练模型并记录学习曲线
for epoch in range(best_params['n_estimators']):
    model.set_params(n_estimators=epoch + 1)  # 每次迭代增加一个树
    model.fit(X_train_selected, y_train, eval_set=evals, verbose=False)  # 不输出详细信息

    # 记录训练集和验证集的 R²
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)

    train_r2 = r2_score(y_train, y_train_pred)
    val_r2 = r2_score(y_test, y_test_pred)

    train_r2_scores.append(train_r2)
    val_r2_scores.append(val_r2)

data = {
'Estimator': range(1, best_params['n_estimators'] + 1),
'Train R²': train_r2_scores,
'Validation R²': val_r2_scores
}
ds = pd.DataFrame(data)
ds.to_excel("F:/LHSH/0004.LSTM/jhy/test0515/LH_XGBoost_r2_training_0519.xlsx", index=False)

# 绘制 R² 学习曲线
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 10
plt.plot(range(1, best_params['n_estimators'] + 1), train_r2_scores, label='Train R²')
plt.plot(range(1, best_params['n_estimators'] + 1), val_r2_scores, label='Validation R²')
plt.xlabel('n_estimators')
plt.ylabel('R² Score')
plt.title('(1) LH_Optuna_XGBoost learning curve')
plt.legend()

# 训练最终模型并计算 R²
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X_train_selected, y_train)

# 在训练集上进行验证
y_train_pred = final_model.predict(X_train_selected)
train_r2 = r2_score(y_train, y_train_pred)
print("Final Train R² score:", train_r2)

# 输出测试集进行验证
y_test_pred = final_model.predict(X_test_selected)
test_r2 = r2_score(y_test, y_test_pred)
print("Final Test R² score:", test_r2)

# 计算测试集 RMSE
rmse = mean_squared_error(y_test, y_test_pred, squared=False)
print("Test RMSE:", rmse)

with open("XGBoost_best_model_LH.pkl", "wb") as f:
    pickle.dump(model, f) # 保存模型

results = pd.DataFrame({'y_test': y_test, 'y_predicted': y_test_pred})
results.to_excel('F:/LHSH/0004.LSTM/jhy/test0515/OPTUNA_XGB_LH_0519.xlsx', index=False)

plt.show()
