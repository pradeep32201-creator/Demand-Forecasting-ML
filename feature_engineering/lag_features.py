
def add_lag_features(df):

    df= df.copy()


    df['lag_1']= df.groupby('variant_id')['quantity'].shift(1)
    df['lag_7']  =df.groupby('variant_id')['quantity'].shift(7)
    df['lag_30']=df.groupby('variant_id')['quantity'].shift(30)

    return df