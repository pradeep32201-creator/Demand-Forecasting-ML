from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np 

def evolution_metrics(y_test, predictions):
    

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    non_zero_mask = y_test !=0
    mape = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask])/ y_test[non_zero_mask]))* 100

    return mae, rmse, mape