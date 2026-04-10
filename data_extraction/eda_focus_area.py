import matplotlib.pyplot as plt
import pandas as pd


def run_eda(df):
    daily_demand = df.groupby('date')['quantity'].sum()
    plt.figure(figsize=(14, 5))
    plt.plot(daily_demand.index, daily_demand.values)
    plt.title('Daily Total Demand')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.tight_layout()
    plt.savefig('output/daily_demand.png')


    df['month'] = pd.to_datetime(df['date']).dt.month
    monthly = df.groupby('month')['quantity'].mean()
    plt.figure(figsize=(10,5))
    monthly.plot(kind='bar')
    plt.title('Average Daily Demand by Month')
    plt.xlabel('Month')
    plt.ylabel('Avg Units')
    plt.tight_layout()


    plt.figure(figsize=(10,5))
    daily_demand.hist(bins=50)
    plt.title('Distribution of Daily Demand')
    plt.xlabel('Units Sold')
    plt.ylabel('Frequency')
    plt.tight_layout()


