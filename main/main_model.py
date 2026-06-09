from data_extraction.data_extraction import load_data
from data_extraction.eda_focus_area import run_eda

df = load_data()
run_eda(df)

from feature_engineering.time_features import add_time_features
from feature_engineering.lag_features import add_lag_features
from feature_engineering.rolling_statistics import rolling_features

df = add_time_features(df)
df = add_lag_features(df)
df = rolling_features(df)


from model_building.train_test_split import train_test_split
from model_building.baseline_model import run_baseline
from model_building.scikit_learn_models import train_rf
from model_building.xgboost import train_xgb



from model_building.prophet_model import run_prophet

from model_building.time_series_cross_validation import time_series_cv

train,test = train_test_split(df)
features = ['day_of_week', 'month', 'lag_1', 'lag_7', 'rolling_mean_7']
X_train = train[features].dropna()
y_train = train.loc[X_train.index, 'quantity']

X_test = test[features].dropna()
y_test = test.loc[X_test.index, 'quantity']

baseline_mae = run_baseline(test)
run_prophet(df)


rf_model = train_rf(X_train, y_train)
predictions = rf_model.predict(X_test)

xgb_model = train_xgb(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)


print("\n-- Random Forest CV --")
time_series_cv(rf_model, X_train, y_train)

print("\n-- XGBoost CV --")
time_series_cv(xgb_model, X_train, y_train)


from model_evaluation.error_metrics import evolution_metrics


rf_mae, rf_rmse, rf_mape = evolution_metrics(y_test, predictions)
xgb_mae, xgb_rmse, xgb_mape = evolution_metrics(y_test, xgb_predictions)

print(f"\n-- Model Comparison -------------------------")
print(f'Baseline (lag-7) MAE : {baseline_mae:.2f}')
print(f"Baseline (lag-7) MAE : {baseline_mae:.2f}")
print(f"Random Forest    MAE : {rf_mae:.2f}  RMSE: {rf_rmse:.2f}  MAPE: {rf_mape:.2f}%")
print(f"XGBoost          MAE : {xgb_mae:.2f}  RMSE: {xgb_rmse:.2f}  MAPE: {xgb_mape:.2f}%")

from model_evaluation.visualization import plot_training_history
import pandas as pd

plot_dates = pd.Series(test['date'].loc[X_test.index]).groupby(test['date'].loc[X_test.index]).first().index
actual_daily = y_test.groupby(test['date'].loc[X_test.index]).sum()
predicted_daily = pd.Series(xgb_predictions, index=X_test.index).groupby(test['date'].loc[X_test.index]).sum()

plot_training_history(actual_daily.index, actual_daily, predicted_daily)

from model_evaluation.residual_analysis import plot_residuals
plot_residuals(y_test, predictions)




from deployment.monitor import monitor_accuracy
monitor_accuracy(predictions, y_test, str(test['date'].max()))

from deployment.model_serialization import save_and_predict
save_and_predict(rf_model,X_test)





