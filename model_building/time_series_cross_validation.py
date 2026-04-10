from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import numpy as np

def time_series_cv(model,X,y):


    tscv = TimeSeriesSplit(n_splits=5)

    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        scores.append(mean_absolute_error(y_te, preds))



    print(f"CV MAE scores: {[round(s, 2) for s in scores]}")

    print(f"Mean MAE: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")

    return scores