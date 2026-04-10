

from data_extraction.data_extraction import load_data
from data_extraction.eda_focus_area import run_eda

df = load_data()
run_eda(df)

from feature_engineering.time_features import create_time_features
from feature_engineering.lag_features import add_lag_features
from feature_engineering.rolling_statistics import rolling_features

df = create_time_features(df)
df = add_lag_features(df)
df = rolling_features(df)


from model_building.train_test_split import test_train_split
from model_building.baseline_model import run_baseline
from model_building.scikit_learn_models import train_rf
from model_building.xgboost import train_xgb



from model_building.prophet import run_prophet
run_prophet(df)

from model_building.time_series_cross_validation import time_series_cv

train,test = test_train_split(df)
run_baseline(test)
features = ['day_of_week', 'month', 'lag_1', 'rolling_mean_7']
X_test = test[features].dropna()
y_test = test.loc[X_test.index, 'quantity']

X_train = train[features].dropna()
y_train = train.loc[X_train.index, 'quantity']

rf_model = train_rf(X_train, y_train)
predictions = rf_model.predict(X_test)



time_series_cv(rf_model, X_train, y_train)
train_xgb(X_train, y_train)

from model_evaluation.error_metrics import evolution_metrics

mae, rmse, mape = evolution_metrics(y_test, predictions)
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}")


from model_evaluation.visualization import plot_training_history
plot_training_history(test['date'].loc[X_test.index], y_test, predictions)

from model_evaluation.residual_analysis import plot_residuals
plot_residuals(y_test, predictions)




from deployment.monitor import monitor_accuracy
monitor_accuracy(predictions, y_test, str(test['date'].max()))

from deployment.model_serialization import save_and_predict
save_and_predict(rf_model,X_test)





