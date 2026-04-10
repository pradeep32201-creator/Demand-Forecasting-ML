from sklearn.metrics import mean_absolute_error


def  run_baseline(test):

    test= test.copy()


    baseline_mask =  test['lag_7'].notna()
    baseline_mae = mean_absolute_error(
        test.loc[baseline_mask, 'quantity'],
        test.loc[baseline_mask, 'lag_7']
    )

    print(f'Baseline (lag-7) MAE: {baseline_mae:.2f}')
    return baseline_mae
