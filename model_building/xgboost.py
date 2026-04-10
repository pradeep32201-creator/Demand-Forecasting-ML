from xgboost import XGBRegressor


def train_xgb(X_train, y_train):
    
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    xgb_model.fit(X_train, y_train)


    
    return xgb_model