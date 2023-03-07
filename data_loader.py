import pandas as pd

def load_data():
    df = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/fluo/fluo_11_32_37_slice1.csv',
                     index_col='Event Index')
    df_features = pd.read_csv(r'/Users/razmigderounian/PycharmProjects/GECIquant_trial/features/features_11_32_37_slice1.csv')
    return df, df_features

