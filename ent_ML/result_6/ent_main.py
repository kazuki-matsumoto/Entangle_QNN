from ent_classification import QclClassification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import time
import copy
import random
from pathlib import Path
from qiskit.utils import algorithm_globals


# データセットの読み込み
forest = datasets.fetch_covtype()
# 量子回路のパラメータ
C_DEPTH = 2 ## circuitの深さ
NUM_CLASS = 4 ## 分類数（ここでは3つの品種に分類）
NUM_FEATCHERS = 4 ## 特徴量の次元
DATA_SIZE = 100 ## 学習データのサイズ
BLOCK_SIZE = 4 ## 同じデータの個数（正解クラスの個数 + 不正解クラスの個数）

DATA_NUM_QUBITS = 2 ## 入力データに使う量子ビット数
CLASS_NUM_QUBITS = 2 ## ラベルに使う量子ビット数
ANCILLA_NUM_QUBITS = 1 ## アンシーラ量子ビット数
N_QUBITS = 2 * (DATA_NUM_QUBITS + CLASS_NUM_QUBITS) + ANCILLA_NUM_QUBITS ## qubitの数

PARAMS_SEED = 0 ## 初期パラメータのシード値
DATA_SEED = 0 ## データセットのランダムのシード値

MAX_ITER = 100  ## パラメータの更新回数

# BATCH_SIZE = 1

FOLDER_PATH = f''
FIG_NAME_LOSS = FOLDER_PATH + 'graph_loss.jpeg'

NUM_EPOCHS = 2

# data sets
def dataframe(dataset, num_class, num_features, data_size):
    
    X = dataset.data[:, [i for i in range(num_features)]]
    X_df = pd.DataFrame(data=X, columns=dataset.feature_names[:num_features])
    y = dataset.target
    y_df = pd.DataFrame(data=y, columns=['target'])

    df = pd.concat([X_df, y_df], axis=1)
    df = df[df['target'] <= num_class]
    
    label_list = df['target'].unique()
    df_tmp = df.loc[:, df.columns[:-1]]
    
    # 正規化
    df.loc[:, df.columns[:-1]] = (df_tmp - df_tmp.min()) / (df_tmp.max() - df_tmp.min())
    
    df_dict = {}
    for name, group in df.groupby('target'):
        df_dict[name] = group.reset_index(drop=True)
        df_dict[name] = df_dict[name].iloc[range(int(data_size / 4)), :]
    
    df = pd.concat([df_dict[n] for n in df['target'].unique()], ignore_index=True)
    
    is_corrects = np.array([True]*data_size)
    is_corrects_df = pd.DataFrame(data=is_corrects, columns=['is_correct'])
    
    df = pd.concat([df, is_corrects_df], axis=1)
    
    # 不正解ラベルを追加する
    mixed_df = pd.DataFrame(columns=df.columns)
    mixed_df['is_correct'] = mixed_df['is_correct'].astype(bool)
        
    for id in range(data_size):
        
        incorrect_df = pd.DataFrame(columns=df.columns)
        tmp_mixed_df = pd.DataFrame(columns=df.columns)
        
        incorrect_df['is_correct'] = incorrect_df['is_correct'].astype(bool)
        tmp_mixed_df['is_correct'] = tmp_mixed_df['is_correct'].astype(bool)
    
        df_at = df[id:id+1]
        correct_label = df_at.iloc[0, -2]
        incorrect_labels = label_list[label_list != correct_label]
        
        ## 不正解クラスが3つのとき
        for label in incorrect_labels:
            
            tmp_df_at = copy.copy(df_at)
            tmp_df_at['target'] = label
            tmp_df_at['is_correct'] = False
            
            # 3つの不正解データフレームの組
            incorrect_df = pd.concat([incorrect_df, tmp_df_at], ignore_index=True)
        
        ## 不正解クラスが1つのとき
        # incorrect_df = copy.copy(df_at)
        # incorrect_df['target'] = random.choice(incorrect_labels)
        # incorrect_df['is_correct'] = False
        
        
        tmp_mixed_df = pd.concat([incorrect_df, df_at], ignore_index=True)
        tmp_mixed_df = tmp_mixed_df.sample(frac=1, ignore_index=True)
        mixed_df = pd.concat([mixed_df, tmp_mixed_df], ignore_index=True)

    filepath = Path('dataframe/csv/mixed_df.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True) 
    mixed_df.to_csv(filepath)
    
    return mixed_df


def plot_data(dataset):
    # データ点のプロット
    plt.figure(figsize=(8, 5))

    for t in range(3):
        x = x_train[dataset.target==t][:,0]
        y = x_train[dataset.target==t][:,1]
        cm = [plt.cm.Paired([c]) for c in [0,6,11]]
        plt.scatter(x, y, c=cm[t], edgecolors='k', label=dataset.target_names[t])

    # label
    plt.title('Iris dataset')
    plt.xlabel('petal length (cm)')
    plt.ylabel('petal width (cm)')
    plt.legend()
    plt.savefig('irisdataset.png')


def train_test_split_from_df(df, test_size, num_class, random_state):
    
    np.random.seed(random_state)
    
    split_shape = BLOCK_SIZE
    
    # 同じデータのブロック
    blocks = [df[i:i+split_shape] for i in range(0, len(df), split_shape)]
    shuffuled_blocks = np.random.permutation(blocks)
    flat_shuffled_blocks = shuffuled_blocks.reshape((-1, shuffuled_blocks.shape[-1]))
    shuffled_df = pd.DataFrame(flat_shuffled_blocks, columns=df.columns)
    
    # データフレームを保存
    filepath = Path('dataframe/csv/shuffled_df.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True) 
    shuffled_df.to_csv(filepath)
    
    # ラベルとデータ
    y = shuffled_df.loc[:, ['target', 'is_correct']].values
    X = shuffled_df.drop('target', axis=1).values

    X_train = X[:(int(len(X)*(1-test_size)) - int(len(X)*(1-test_size) % split_shape))]
    X_test = X[(int(len(X)*(1-test_size)) - int(len(X)*(1-test_size) % split_shape)):]
    y_train = y[:(int(len(y)*(1-test_size)) - int(len(y)*(1-test_size) % split_shape))]
    y_test = y[(int(len(y)*(1-test_size)) - int(len(y)*(1-test_size) % split_shape)):]
    
    is_correct_train = X_train[:, -1]

    # 同じデータについてブロックに分割する
    # X_train = X_train[X_train[:, -1] == True][:, :-1]
    X_train = X_train[:, :-1]
    
    # 不正解クラスを含んだy_train（ワンホット）
    y_train_mixed = np.eye(num_class)[list(y_train[:, :-1].flatten() - 1)]
    y_train_mixed = np.concatenate((y_train_mixed, is_correct_train[:, np.newaxis]), axis=1)

    # 正解クラスのみのy_train（ワンホット）
    y_train_tmp = copy.deepcopy(y_train)
    y_train_tmp = y_train_tmp[y_train_tmp[:, -1] == True]
    y_train = np.eye(num_class)[list(y_train_tmp[:, :-1].flatten() - 1)]

    # テストクラスは、正解クラスのみ
    y_test = y_test[y_test[:, -1] == True][:, :-1].flatten()
    y_test = np.eye(num_class)[list(y_test - 1)]
    X_test = X_test[X_test[:, -1] == True][:, :-1]    
    
    return X_train, X_test, y_train, y_train_mixed, y_test
    

if __name__ == '__main__':

    ## データ作成
    df = dataframe(forest, NUM_CLASS, NUM_FEATCHERS, DATA_SIZE)

    ## 学習データ・テストデータの作成
    X_train, X_test, y_train, y_train_mixed, y_test = train_test_split_from_df(
        df, test_size=0.3, num_class=NUM_CLASS, random_state=DATA_SEED
    )
    
    # print('X_train:\n', X_train)
    # print('X_train:\n', X_train.shape)
    # print('X_test:\n', X_test)
    # print('y_train:\n', y_train)
    # print('y_train_mixed:\n', y_train_mixed)
    # print('y_train_mixed.shape:\n', y_train_mixed.shape)
    # print('y_test:\n', y_test)
    
    BATCH_SIZE = X_train.shape[0]
    
    filepath_loss = Path(FIG_NAME_LOSS)
    filepath_loss.parent.mkdir(parents=True, exist_ok=True)

    # 乱数発生器の初期化（量子回路のパラメータの初期値に用いる）
    algorithm_globals.random_seed = PARAMS_SEED

    # QclClassificationクラスをインスタンス化
    qcl = QclClassification(N_QUBITS, DATA_NUM_QUBITS, CLASS_NUM_QUBITS, NUM_FEATCHERS, C_DEPTH, NUM_CLASS, FIG_NAME_LOSS, NUM_EPOCHS, BATCH_SIZE)

    res, theta_init, theta_opt = qcl.fit(X_train, y_train, y_train_mixed, maxiter=MAX_ITER)
    
    # accuracy = qcl.accuracy_score(theta_opt, X_test, y_test)
    # print(accuracy)
