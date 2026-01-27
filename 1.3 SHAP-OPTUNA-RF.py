import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import RFE
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
# 0.0 read input
df = pd.read_excel('F:/LHSH/0004.LSTM/jhy/LH_site_temp_smooth_0123.xlsx',engine='openpyxl', index_col=0) #, index_col=0)
df = df.iloc[:, 2:]
X=df
X = X.drop('OBS_lh_QC', axis=1)
X = X.drop('OBS_lh', axis=1)
X = X.iloc[:, 0:-1]
y= df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1.0 Define the Optuna objective function, using R² as the objective.
def objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", low=10, high=50),
        "max_depth": trial.suggest_int("max_depth", low=5, high=15),
        "min_samples_split": trial.suggest_int("min_samples_split", low=5, high=10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", low=5, high=10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
        "max_samples": trial.suggest_float("max_samples", 0.7, 0.95),
        "random_state": 42
    }
    model = RandomForestRegressor(**param)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    return scores.mean()


study = optuna.create_study(direction="maximize")  # maximize R²
study.optimize(objective, n_trials=15)

print("Best trial:")
print(study.best_trial.params)
# -----Train the model using optimal parameters and make predictions.
best_params = study.best_trial.params
model = RandomForestRegressor(**best_params)

# ----2.0-----Feature Selection Using RFE
n_features = X_train.shape[1]
n_features_to_select = min(12, n_features)
selector = RFE(model, n_features_to_select=n_features_to_select)
selector = selector.fit(X_train, y_train)
selected_features = X_train.columns[selector.support_]
#
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

X_test_selected_df = pd.DataFrame(X_test_selected, columns=selected_features)
# Save X_test_selected as a new Excel file
# X_test_selected_df.to_excel('OPTUNA_RF_LH_X_test_selector_0519.xlsx', index=False)

evals = [(X_train_selected, y_train), (X_test_selected, y_test)]
evals_result = {}

train_r2_scores = []
val_r2_scores = []

# --3.0---- Train the model and record the learning curve
for epoch in range(best_params['n_estimators']):
    model.set_params(n_estimators=epoch + 1)
    model.fit(X_train_selected, y_train)

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
# ds.to_excel("test0515/LH_RF_r2_training0519.xlsx", index=False)


# ----3.1----Train the final model and calculate R².
final_model = RandomForestRegressor(**best_params)
final_model.fit(X_train_selected, y_train)

y_train_pred = final_model.predict(X_train_selected)
train_r2 = r2_score(y_train, y_train_pred)
print("Final Train R² score:", train_r2)

y_test_pred = final_model.predict(X_test_selected)
test_r2 = r2_score(y_test, y_test_pred)
print("Final Test R² score:", test_r2)

rmse = mean_squared_error(y_test, y_test_pred, squared=False)
print("Test RMSE:", rmse)
results = pd.DataFrame({'y_test': y_test, 'y_predicted': y_test_pred})
results.to_excel('OPTUNA_RF_LH_0519.xlsx', index=False)

train_df = pd.DataFrame({
    'OBS': y_train,
    'PRED': y_train_pred,
    'style': 'train'
})

test_df = pd.DataFrame({
    'OBS': y_test,
    'PRED': y_test_pred,
    'style': 'test'
})
results= pd.concat([train_df, test_df], ignore_index=True)
# results.to_excel('train_xgboost_0715/OPTUNA_RF_LH_0715.xlsx', index=False)

with open("RF_best_model_LH.pkl", "wb") as f:
    pickle.dump(model, f) # save the model

# --4.0----Plot the R² learning curve
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 10
plt.plot(range(1, best_params['n_estimators'] + 1), train_r2_scores, label='Train R²')
plt.plot(range(1, best_params['n_estimators'] + 1), val_r2_scores, label='Validation R²')
plt.xlabel('n_estimators')
plt.ylabel('R² Score')
plt.title('(1) LH_Optuna_RF learning curve')
plt.legend()
plt.show()