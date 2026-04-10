from prophet import Prophet
import matplotlib.pyplot as plt 
from pathlib import Path

def run_prophet(df):

    Path('outputs').mkdir(exist_ok=True)


    prophet_df = df.groupby('date')['quantity'].sum().reset_index()

    prophet_df.columns = ['ds', 'y']

    model = Prophet(yearly_seasonality= True, weekly_seasonality=True)

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=30)

    forecast = model.predict(future)

    model.plot(forecast)

    plt.title('Prophet Forecast')
    plt.tight_layout()
    plt.savefig('output/prophet_forecast.png')
    plt.close()
    
    return forecast 
