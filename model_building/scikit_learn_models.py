
from sklearn.ensemble import RandomForestRegressor

#from sklearn.linear_model  import LinearRegression

def train_rf(X_train, y_train):
   # features = ['day_of_week', 'month', 'lag_1', 'lag_7', 'rolling_mean_7']

    #X_train = train[features].dropna()

    #y_train = train.loc[X_train.index,'quantity']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model