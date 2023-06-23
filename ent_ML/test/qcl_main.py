from qcl_classification import QclClassification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import time
from sklearn.model_selection import train_test_split

# Irisデータセットの読み込み
iris = datasets.load_iris()
# 量子回路のパラメータ
nqubit = 3 ## qubitの数。必要とする出力の次元数よりも多い必要がある
c_depth = 2 ## circuitの深さ
num_class = 3 ## 分類数（ここでは3つの品種に分類）


def dataframe(dataset):
    # 扱いやすいよう、pandasのDataFrame形式に変換
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    df['target_names'] = dataset.target_names[dataset.target]
    
    return df

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

def plot_accuracy(x_list, y_list, filename, nqubit):
    plt.plot(x_list, y_list, label='{0} qubit accuracy'.format(nqubit))
    plt.title('accuracy vs nqubits')
    plt.xlabel('n_qubits')
    plt.ylabel('accuracy')
    plt.legend()
    
    plt.savefig('accuracy2nqubits/{0}.png'.format(filename))


## 教師データ作成
# ここではpetal length, petal widthの2種類のデータを用いる。
df = dataframe(iris)
X = df.loc[:,['petal length (cm)', 'petal width (cm)']].to_numpy() # shape:(150, 2)
y = np.eye(3)[iris.target] # one-hot 表現 shape:(150, 3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2)

# 乱数発生器の初期化（量子回路のパラメータの初期値に用いる）
random_seed = 0
np.random.seed(random_seed)
accuracies = np.array([])

x_list = np.array([])

np.append(x_list, nqubit)
# QclClassificationクラスをインスタンス化
qcl = QclClassification(nqubit, c_depth, num_class)

res, theta_init, theta_opt = qcl.fit(X_train, y_train, maxiter=10)

# accuracy = qcl.accuracy_score(theta_opt, X_test, y_test)
# np.append(accuracies, accuracy)
# plot_accuracy(x_list=x_list, y_list=accuracies, filename='{0}qubit'.format(nqubit), nqubit=nqubit)
