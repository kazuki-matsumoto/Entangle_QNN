from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split

def dataset():
    data = load_iris()
    print(data.target)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df = df.iloc[:,:2]
    df['target'] = data.target
 
    print(df)

dataset()