from sklearn.metrics import mean_absolute_error

default_threshold = 50

def monitor_accuracy(predictions, actuals, date, threshold=default_threshold):

    error = mean_absolute_error(actuals, predictions)
    #log_metric('forecast_mae', error, date)
    print(f'[{date}] MAE: {error:.2f}')

    if error > threshold:
        print(f'⚠️ Model performance degraded - MAE {error:.2f} exceeds threshold {threshold}')
    else:
        print(f'✅ Model within acceptable range.')

