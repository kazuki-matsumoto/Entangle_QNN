from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

NUM_CLASS = 4
NUM_FEATCHERS = 4
DATA_SIZE = 200

# data sets
def datasets(num_class, num_feachers, data_size):
    
    forest = fetch_covtype()

    X = forest.data[:, [i for i in range(num_feachers)]]
    X_df = pd.DataFrame(data=X, columns=forest.feature_names[:num_feachers])
    y = forest.target
    y_df = pd.DataFrame(data=y, columns=['target'])

    df = pd.concat([X_df, y_df], axis=1)
    df = df[df['target'] <= num_class]

    df_tmp = df.loc[:, df.columns[:-1]]
    # 正規化
    df.loc[:, df.columns[:-1]] = (df_tmp - df_tmp.min()) / (df_tmp.max() - df_tmp.min())
    
    df_dict = {}
    
    for name, group in df.groupby('target'):
        print(name)
        df_dict[name] = group.reset_index(drop=True)
        df_dict[name] = df_dict[name].iloc[range(int(data_size / 4)), :]
    
    df = pd.concat([df_dict[n] for n in df['target'].unique()], ignore_index=True)
    df = df.sample(frac=1, ignore_index=True)
    
    return df

df = datasets(NUM_CLASS, NUM_FEATCHERS, DATA_SIZE)

print(df)

print(df['target'].unique())

print(df['target'].value_counts())