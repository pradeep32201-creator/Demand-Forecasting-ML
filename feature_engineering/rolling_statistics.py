def rolling_features(df):



    df= df.copy()

    df['rolling_mean_7'] = df.groupby('variant_id')['quantity'].transform(lambda x: x.shift(1).rolling(7).mean()
                                                                          
                                                                          
                                                                          
    )

    df['rollimg_std_7']    = df.groupby('variant_id')['quantity'].transform(lambda x: x.shift(1).rolling(7).std())


    return df