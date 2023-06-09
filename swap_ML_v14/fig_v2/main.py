from qiskit import *
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from sklearn.datasets import fetch_covtype
import pandas as pd
import numpy as np
from qiskit.quantum_info import partial_trace, entropy
from sklearn.metrics import accuracy_score
import os
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit import ParameterVector
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import COBYLA, ADAM
import matplotlib.pyplot as plt
import copy
from pathlib import Path



DATA_NUM_QUBITS = 4
CLASS_NUM_QUBITS = 2
ANCILLA_NUM_QUBITS = 1
N_QUBITS = 2 * (DATA_NUM_QUBITS + CLASS_NUM_QUBITS) + ANCILLA_NUM_QUBITS

NUM_CLASS = 4
NUM_FEATCHERS = 4
DATA_SIZE = 100

N_PARAMS = 12 * (DATA_NUM_QUBITS + CLASS_NUM_QUBITS - 1)
N_EPOCH = 10
LEARNING_RATE = 5
BLOCK_SIZE = 4

# Set seed for random generators
algorithm_globals.random_seed = 42
DATA_SEED = 1

FOLDER_PATH = 'fig_v3/'
FIG_NAME_LOSS = FOLDER_PATH + 'graph_loss.jpeg'
FIG_NAME_ACCURACY = FOLDER_PATH + 'graph_accuracy.jpeg'


###############################################################

# data sets
def datasets(num_class, num_features, data_size):
    
    forest = fetch_covtype()
    
    X = forest.data[:, [i for i in range(num_features)]]
    X_df = pd.DataFrame(data=X, columns=forest.feature_names[:num_features])
    y = forest.target
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
        
        for label in incorrect_labels:
            
            tmp_df_at = copy.copy(df_at)
            tmp_df_at['target'] = label
            tmp_df_at['is_correct'] = False
            
            # 3つの不正解データフレームの組
            incorrect_df = pd.concat([incorrect_df, tmp_df_at], ignore_index=True)
            
        tmp_mixed_df = pd.concat([incorrect_df, df_at], ignore_index=True)
        tmp_mixed_df = tmp_mixed_df.sample(frac=1, ignore_index=True)
        mixed_df = pd.concat([mixed_df, tmp_mixed_df], ignore_index=True)

    filepath = Path('dataframe/csv/mixed_df.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True) 
    mixed_df.to_csv(filepath)
    
    return mixed_df

###############################################################


# メインの量子回路
class SwapQNN:
    def __init__(self, nqubits, simulator, nshots, X_train: list, y_train: int, y_mixed_train: list):
        self.nqubits = nqubits
        self.simulator = simulator
        self.nshots = nshots
        self.n_params = N_PARAMS
        
        self.x_train = X_train
        self.y_train = y_train
        self.y_mixed_train = y_mixed_train
        
        self.weights = ParameterVector('weight', self.n_params)
        
    def U_in(self, X, y):
        qr = QuantumRegister(self.nqubits, 'qr')
        cr = ClassicalRegister(1, 'cr')
        qc = QuantumCircuit(qr, cr)
        
        index_y = np.argmax(y)

        # encode for label（y）
        for i, y_i in enumerate(format(index_y, '02b')):
            if y_i == "1":
                qc.x(i)
                qc.x(i + DATA_NUM_QUBITS + CLASS_NUM_QUBITS)
        
        # encode for input data
        for i, x in enumerate(X):
            angle_y = np.arcsin(x)
            angle_z = np.arccos(x**2)
            
            for i in range(CLASS_NUM_QUBITS, DATA_NUM_QUBITS + CLASS_NUM_QUBITS):
                qc.ry(angle_y, i)
                qc.rz(angle_z, i)
                
                qc.ry(angle_y, i + DATA_NUM_QUBITS + CLASS_NUM_QUBITS)
                qc.rz(angle_z, i + DATA_NUM_QUBITS + CLASS_NUM_QUBITS)
        
        return qc, qr
    
    # メインの量子回路
    def U_out(self, qc, qr):
        
        # qc.append(self.grover(), [qr[i] for i in range(5)])
        # qc.append(self.classify_rot_gate(n_qubits=4, state_phase=self.state_phase), [qr[i] for i in [5, 6, 9, 10]])
        qc.append(self.parametarized_qcl(), [qr[i] for i in range(0, CLASS_NUM_QUBITS + DATA_NUM_QUBITS)])
        qc.append(self.max_entanglement_gate(), [qr[i] for i in range(CLASS_NUM_QUBITS + DATA_NUM_QUBITS, self.nqubits-1)])
        
        qc.h(self.nqubits-1)
        
        # swap test
        for id in range(CLASS_NUM_QUBITS + DATA_NUM_QUBITS):
            qc.cswap(self.nqubits-1, self.nqubits-2-id, self.nqubits-2-(CLASS_NUM_QUBITS + DATA_NUM_QUBITS)-id)

        qc.h(self.nqubits-1)
        
        return qc

    def identity_interpret(self, x):
        return x
    
    def qcl_pred(self, X, y):
        qc, qr = self.U_in(X, y)
        qc = self.U_out(qc, qr)
        qc.measure(self.nqubits-1, 0)
        
        qnn = SamplerQNN(
            circuit=qc,
            weight_params=self.weights,
            interpret=self.identity_interpret,
            output_shape=2
        )
        
        return qnn

    def add_paramed_ent_gate(self, qc, id_qubit, skip_id1, skip_id2):
        qc.u(self.weights[id_qubit*12], self.weights[id_qubit*12+1], self.weights[id_qubit*12+2], id_qubit + skip_id1)
        qc.u(self.weights[id_qubit*12+3], self.weights[id_qubit*12+4], self.weights[id_qubit*12+5], id_qubit + skip_id2)
        qc.cx(id_qubit + skip_id1, id_qubit + skip_id2)
        qc.u(self.weights[id_qubit*12+6], self.weights[id_qubit*12+7], self.weights[id_qubit*12+8], id_qubit + skip_id1)
        qc.u(self.weights[id_qubit*12+9], self.weights[id_qubit*12+10], self.weights[id_qubit*12+11], id_qubit + skip_id2)

    def add_paramed_gate(self, qc, id_qubit, skip_id):
        qc.u(self.weights[id_qubit*12 + 12*3], self.weights[id_qubit*12+1 + 12*3], self.weights[id_qubit*12+2 + 12*3], id_qubit + skip_id)
        qc.u(self.weights[id_qubit*12+3 + 12*3], self.weights[id_qubit*12+4 + 12*3], self.weights[id_qubit*12+5 + 12*3], id_qubit + skip_id)
        qc.u(self.weights[id_qubit*12+6 + 12*3], self.weights[id_qubit*12+7 + 12*3], self.weights[id_qubit*12+8 + 12*3], id_qubit + skip_id)
        qc.u(self.weights[id_qubit*12+9 + 12*3], self.weights[id_qubit*12+10 + 12*3], self.weights[id_qubit*12+11 + 12*3], id_qubit + skip_id)

    def label_param(self, n_qubits):
        label_para_qr = QuantumRegister(n_qubits)
        label_para_qc = QuantumCircuit(label_para_qr, name='Label Params Gate')
        
        # パラメータ化回転ゲート
        for i in range(2):
            label_para_qc.rz(self.weights[12*3 + 2*i], i)
            label_para_qc.ry(self.weights[12*3 + 2*i + 1], i)
            
        inst_qc = label_para_qc.to_instruction()
        
        return inst_qc

    # パラメータ化量子回路
    def parametarized_qcl(self):
        
        n_qubits = CLASS_NUM_QUBITS + DATA_NUM_QUBITS
        
        para_ent_qr = QuantumRegister(n_qubits)
        para_ent_qc = QuantumCircuit(para_ent_qr, name="Parametarized Gate")

        for id_qubit in range(n_qubits-1):
            
            self.add_paramed_ent_gate(para_ent_qc, id_qubit, 0, 1)
            
            para_ent_qc.barrier()
        
        inst_paramed_gate = para_ent_qc.to_instruction()
        
        return inst_paramed_gate
    
    # 最大エンタングルメント状態
    def max_entanglement_gate(self):
        
        n_qubits = CLASS_NUM_QUBITS + DATA_NUM_QUBITS
        
        max_ent_qr = QuantumRegister(n_qubits)
        max_ent_qc = QuantumCircuit(max_ent_qr, name="Max Entanglement Gate")
        
        max_ent_qc.h(0)
        
        for i in range(n_qubits-1):
            max_ent_qc.cx(i, i+1)
        
        inst_max_ent = max_ent_qc.to_instruction()
        return inst_max_ent
    
    # コスト関数（正解ラベル）
    def cost_func(self, weights):
        
        innerproducts = np.zeros(BLOCK_SIZE)
        losses = np.zeros(BLOCK_SIZE)
        
        # 測定
        for i, (x, y_mixed) in enumerate(zip(self.x_train, self.y_mixed_train)):
        
            qnn = self.qcl_pred(x, y_mixed)
            probabilities = qnn.forward(input_data=None, weights=weights)        
            # print('cost probabilities', probabilities)
            
            # 0 <= np.sum(probabilities[:, 1]) <= 0.5
            innerproducts[i] = np.sum(probabilities[:, 1])      
            # print('innerproducts', innerproducts)
            
        softmaxed_innerproducts = softmax(innerproducts)
        
        print('self.y_train', self.y_train)
        
        # 2乗和誤差
        LOSS = 0.5 * np.sum((softmaxed_innerproducts - self.y_train)**2)
        
        # クロスエントロピー誤差
        # delta = 1e-7
        # LOSS = - np.sum(self.y_train * np.log(softmaxed_innerproducts + delta))
        
        return LOSS
    
    # 勾配計算
    def calc_gradient(self, params):
        grad = np.zeros_like(params)
        
        for i in range(len(params)):
            shifted = params.copy()
            shifted[i] += np.pi/2
            forward = self.cost_func(shifted)
            # print("forward", forward)
            
            shifted[i] -= np.pi
            backward = self.cost_func(shifted)
            # print("backward", backward)
            
            grad[i] = 0.5 * (forward - backward)
        
        return np.round(grad, 10)
        
    
    # パラメータ更新
    def update_weights(self, weights):

        grad = self.calc_gradient(weights)
        print("grad", grad)
        print("weights", weights)
        updated_weights = weights - LEARNING_RATE * grad
        
        return updated_weights
    
    
    def predict(self, x_test, optimed_weight):
        innerproducts = np.array([])
        
        print('----------------------------- predict -----------------------------')
        
        test_classes = np.array(range(NUM_CLASS))
        
        for y in np.eye(NUM_CLASS)[test_classes]:
            qnn = self.qcl_pred(X=x_test, y=y)
            probabilities = qnn.forward(input_data=None, weights=optimed_weight)
            print("probabilities", probabilities)
            innerproducts = np.append(innerproducts, np.sum(probabilities[:, 1]))
        
        print("innerproducts", innerproducts)
        
        return np.argmax(innerproducts) + 1


    def accuracy(self, X_test, y_test, optimed_weight):

        y_pred = np.array([])
        
        for x in X_test:
            predicted = self.predict(x_test=x, optimed_weight=optimed_weight)
            y_pred = np.append(y_pred, predicted)
        
        print('y_pred', y_pred)
        # print('y_pred.shape', y_pred.shape)
        # print('y_test', y_test)
        # print('y_test.shape', y_test.shape)

        print("accuracy_score :", accuracy_score(list(y_test), list(y_pred)))
        
        accuracy = accuracy_score(list(y_test), list(y_pred))
        
        graph_accuracy(accuracy, title="Accuracy value against iteration")
        
        return accuracy

def softmax(x):
    x = x - np.max(x, axis=0)
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# ロスのグラフ（点はラベル）
def graph_loss(loss, y, title):
    fig1, ax1 = plt.subplots()
    
    y_list.append(np.argmax(y) + 1)
    print('y_list: ', y_list)

    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("LOSS value")
    
    loss_func_vals.append(loss)
    ax1.plot(range(len(loss_func_vals)), loss_func_vals)
    
    cnt_1 = 0
    cnt_2 = 0
    cnt_3 = 0
    cnt_4 = 0
    
    for (i, point), y in zip(enumerate(loss_func_vals), y_list):
        if y == 1:
            ax1.plot(i, point, '.', markersize=10, color='red', label='y={0}'.format(y) if cnt_1==0 else "")
            cnt_1 += 1
        if y == 2:
            ax1.plot(i, point, '.', markersize=10, color='blue', label='y={0}'.format(y) if cnt_2==0 else "")
            cnt_2 += 1
        if y == 3:
            ax1.plot(i, point, '.', markersize=10, color='green', label='y={0}'.format(y) if cnt_3==0 else "")
            cnt_3 += 1
        if y == 4:
            ax1.plot(i, point, '.', markersize=10, color='yellow', label='y={0}'.format(y) if cnt_4==0 else "")
            cnt_4 += 1
    
    ax1.legend()
    plt.savefig(FIG_NAME_LOSS)
    plt.close()


def graph_accuracy(accuracy, title):
    fig2, ax2 = plt.subplots()
    accuracy_vals.append(accuracy)
    ax2.set_title(title)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy value")
    ax2.plot(range(len(accuracy_vals)), accuracy_vals)
    plt.savefig(FIG_NAME_ACCURACY)
    plt.close()
    
def save_file_at_dir(dir_path, filename, dataframe):
    os.makedirs(dir_path, exist_ok=True)
    dataframe.to_csv(dir_path+filename)

def train_test_split(df, test_size, random_state):
    np.random.seed(random_state)
    
    split_shape = 4
    
    blocks = [df[i:i+4] for i in range(0, len(df), 4)]
    shuffuled_blocks = np.random.permutation(blocks)
    flat_shuffled_blocks = shuffuled_blocks.reshape((-1, shuffuled_blocks.shape[-1]))
    shuffled_df = pd.DataFrame(flat_shuffled_blocks, columns=df.columns)
    
    filepath = Path('dataframe/csv/shuffled_df.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True) 
    shuffled_df.to_csv(filepath)
    
    y = shuffled_df.loc[:, ['target', 'is_correct']].values
    X = shuffled_df.drop('target', axis=1).values

    X_train = X[:(int(len(X)*(1-test_size)) - int(len(X)*(1-test_size) % split_shape))]
    X_test = X[(int(len(X)*(1-test_size)) - int(len(X)*(1-test_size) % split_shape)):]
    y_train = y[:(int(len(y)*(1-test_size)) - int(len(y)*(1-test_size) % split_shape))]
    y_test = y[(int(len(y)*(1-test_size)) - int(len(y)*(1-test_size) % split_shape)):]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # simlator
    simulator = AerSimulator(device='GPU')
    # num shots
    nshots = 100000
    
    loss_func_vals = []
    accuracy_vals = []
    
    loss_point = []
    y_list = []
    
    filepath_loss = Path(FIG_NAME_LOSS)
    filepath_accuracy = Path(FIG_NAME_ACCURACY)
    filepath_loss.parent.mkdir(parents=True, exist_ok=True)
    filepath_accuracy.parent.mkdir(parents=True, exist_ok=True)
    
    if NUM_CLASS <= 4:
        
        df_dict = {}
        df = datasets(NUM_CLASS, NUM_FEATCHERS, DATA_SIZE)

        X_train, X_test, y_train, y_test = train_test_split(
            df, test_size=0.3, random_state=DATA_SEED
        )

        y = df["target"].values
        # y_list = list(set(y))
        
        n_labels = len(np.unique(y))
        y_train_mixed_onehot = np.eye(n_labels)[list(y_train[:, :-1].flatten() - 1)].reshape(int(y_train[:, :-1].flatten().shape[0]/BLOCK_SIZE), BLOCK_SIZE, -1)
        
        y_train_tmp = copy.deepcopy(y_train)
        y_train_tmp = y_train_tmp[y_train_tmp[:, -1] == True]
        y_train_onehot = np.eye(n_labels)[list(y_train_tmp[:, :-1].flatten() - 1)]
        y_test = y_test[y_test[:, -1] == True][:, :-1].flatten()
        
        X_train = X_train[:, :-1].reshape(int(X_train.shape[0]/BLOCK_SIZE), BLOCK_SIZE, -1)
        X_test = X_test[X_test[:, -1] == True][:, :-1]

        
        initial_weights = 0.1 * (2 * algorithm_globals.random.random(N_PARAMS) - 1)
        optimized_weight = initial_weights
        
        # 学習
        for epoch in range(N_EPOCH):
            print("epoch : ", epoch+1)
            print("---------------------------------- {} epoch ----------------------------------".format(epoch+1))
            
            for x, y, y_mixed in zip(X_train, y_train_onehot, y_train_mixed_onehot):
                print("x", x)
                print('y', y)
                print('y_mixed', y_mixed)
 
                qc = SwapQNN(nqubits=N_QUBITS, simulator=simulator, nshots=nshots, X_train=x, y_train=y, y_mixed_train=y_mixed)
                
                # 最適化
                optimized_weight = qc.update_weights(optimized_weight)
                
                graph_loss(qc.cost_func(optimized_weight), y, title="Objective function value against iteration")
                
                # 推論
                qc.accuracy(X_test, y_test, optimized_weight)


