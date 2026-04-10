
import pandas as pd


def add_time_features(df):

    df= df.copy()
    df['date'] = pd.to_datetime(df['date'])

    
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['day_of_month'] = df['date'].dt.day

    return df

