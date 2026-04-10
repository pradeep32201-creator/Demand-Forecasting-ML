
import pandas as pd

def train_test_split(df):
    


    split_date= df['date'].quantile(0.8)
    train = df[df['date']< split_date] 
    test = df[df['date']>= split_date]
    print(f'Train:{train["date"].min()} to {train["date"].max()}')
    print(f'Test:{test["date"].min()} to {test["date"].max()}')

    return train, test