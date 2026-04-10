import joblib
from pathlib import Path


def save_and_predict(model, X_test):

    Path('models').mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, 'models/demand_model.joblib')
    print(f"Model saved.Size: {Path('models/demand_model.joblib').stat().st_size/1024:.0f} KB")

    loaded_model = joblib.load('models/demand_model.joblib')
    predictions = loaded_model.predict(X_test)

    return predictions