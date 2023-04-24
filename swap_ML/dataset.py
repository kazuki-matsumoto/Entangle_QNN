from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

NUM_CLASS = 4
NUM_FEATCHERS = 4
DATA_SIZE = 100

def datasets(num_class, num_feachers):

    mns = MinMaxScaler()
    forest = fetch_covtype()

    X = forest.data[:, [i for i in range(num_feachers)]]
    X_df = pd.DataFrame(data=X, columns=forest.feature_names[:num_feachers])
    y = forest.target
    print(set(y))
    y_df = pd.DataFrame(data=y, columns=['target'])
    print(X_df)
    print(y_df)

    df = pd.concat([X_df, y_df], axis=1)
    df = df[df['target'] <= num_class]
    df = df.sort_values(by='target').reset_index(drop=True)
    print(df)
    df_tmp = df.loc[:, df.columns[:-1]]
    df.loc[:, df.columns[:-1]] = (df_tmp - df_tmp.min()) / (df_tmp.max() - df_tmp.min())
    print(df)

    tmp = set(df['target'].to_list())
    print(tmp)
    
    return df


df = datasets(NUM_CLASS, NUM_FEATCHERS)
df_dict = {}
for name, group in df.groupby('target'):
    print("name", name)
    df_dict[name] = group.reset_index(drop=True)
    df_dict[name] = df_dict[name].iloc[range(DATA_SIZE), :]

for label, df_split in df_dict.items():   
    print("label", label)
    X = df_split.drop('target', axis=1).values

for k, v in df_dict.items():
    print("k : ", k)
    print("v : ", v)

X = df.drop('target', axis=1).values
print(X)
# for x in X:
#     print(x)

# print(X)
# print(df)
# print(df_dict)
