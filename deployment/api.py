from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from sqlalchemy import create_engine
from datetime import timedelta
app = FastAPI()
model = joblib.load('models/demand_model.joblib')

conn_str = 'postgresql://ecom_ro_user.imnzftquwjuxcwpeufwp:work-experience-read-only@aws-0-ap-south-1.pooler.supabase.com:5432/postgres?sslmode=require'

def fetch_recent_actuals(variant_id:int, reference_date:pd.Timestamp) -> dict:
    """Query the last 7 days of actuals for a variant to compute real lag features."""
    engine= create_engine(conn_str)
    start = reference_date - timedelta(days=7)

    query = f"""
        select date(created_at) as date, sum(qty) as quantity
        from ecom.orders o
        join ecom.order_items oi on o.order_id = oi.order_id
        where variant_id = {variant_id}
          and date(created_at) between '{start.date()}' and '{reference_date.date()}'
        group by date(created_at)
        order by date
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df


def build_features(date, variant_id) -> pd.DataFrame:
    ref_date = pd.to.datetime(date)
    recent = fetch_recent_actuals(variant_id, ref_date)

    lag_1_row = recent[recent['date'] == (ref_date - timedelta(days=1)).date()]
    lag_1 = float(lag_1_row['quantity'].mean()) if not recent.empty else 0.0

    rolling_mean_7 = float(recent['quantity'].mean()) if not recent.empty else 0.0

    df = pd.DataFrame({
        'date': [ref_date],
        'day_od_week': [ref_date.dayofweek],
        'month': [ref_date.month],
        'lag_1': [lag_1],
        'rolling_mean_7': [rolling_mean_7],
    })

    return df[['day_of_week', 'month', 'lag_1', 'rolling_mean_7']]

@app.get("/")

def home():
    return {"message": "Demand Forecasting API is running 🚀"}

@app.post("/predict")

def predict(date: str, variant_id: int):

    try:

        features = build_features(date, variant_id)
    

        predictions = model.predict(features)[0]

    
    
        return {
        'date':date,
        'variant_id': variant_id,
        'predictions': float(predictions)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))