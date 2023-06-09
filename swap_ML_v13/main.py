from qiskit import *
import numpy as np
from scipy.optimize import minimize
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
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


NUM_CLASS = 4
NUM_FEATCHERS = 4
DATA_SIZE = 100
# N_PARAMS = 12*3 + 12*4
N_PARAMS = 12*3
MAX_ITER = 500
N_EPOCH = 10
N_QUBITS = 9
LEARNING_RATE = 0.001

OPTIM_STEPS = 100

# Set seed for random generators
PARAMS_SEED = 1
# algorithm_globals.random_seed = PARAMS_SEED

DATA_SEED = 1

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

    filepath = Path('fig_v4/dataframe/mixed_df.csv') 
    filepath.parent.mkdir(parents=True, exist_ok=True) 
    mixed_df.to_csv(filepath)
    
    return mixed_df

###############################################################


# メインの量子回路
class SwapQNN:
    def __init__(self, nqubits, simulator, nshots, X_train: list, y_train: int):
        self.nqubits_train = nqubits
        self.simulator = simulator
        self.nshots = nshots
        self.n_params = N_PARAMS
        
        self.x_train = X_train
        self.y_train = y_train
        
        self.weights = ParameterVector('weight', self.n_params)
        
    def U_in(self, X, y):
        qr = QuantumRegister(self.nqubits_train, 'qr')
        cr = ClassicalRegister(1, 'cr')
        qc = QuantumCircuit(qr, cr)
        
        # len(y_bin) == 2
        y_bin = format(y, '02b')
        
        # encode for label
        for i, y in enumerate(y_bin):
            if y == "1":
                qc.x(i)
                qc.x(i+4)
        
        # qc.append(self.label_param(2), [qr[4], qr[5]])
        
        # encode for input data
        for x in X:
            angle_y = np.arcsin(x)
            angle_z = np.arccos(x**2)
            
            for i in range(2, 2+2):
                qc.ry(angle_y, i)
                qc.rz(angle_z, i)
                
                qc.ry(angle_y, i+4)
                qc.rz(angle_z, i+4)
        
        # print(qc)
        
        return qc, qr
    
    # メインの量子回路
    def U_out(self, qc, qr):
        
        # qc.append(self.grover(), [qr[i] for i in range(5)])
        # qc.append(self.classify_rot_gate(n_qubits=4, state_phase=self.state_phase), [qr[i] for i in [5, 6, 9, 10]])
        qc.append(self.max_entanglement_gate(), [qr[i] for i in range(4, self.nqubits_train-1)])
        qc.append(self.parametarized_qcl(), [qr[i] for i in range(0, 4)])
        
        qc.h(self.nqubits_train-1)
        
        # swap test
        for id in range(4):
            qc.cswap(self.nqubits_train-1, self.nqubits_train-2-id, self.nqubits_train-6-id)

        qc.h(self.nqubits_train-1)
        
        return qc

    def identity_interpret(self, x):
        return x
    
    def qcl_pred(self, X, y):
        qc, qr = self.U_in(X, y)
        qc = self.U_out(qc, qr)
        qc.measure(self.nqubits_train-1, 0)
        
        qnn = SamplerQNN(
            circuit=qc,
            weight_params=self.weights,
            interpret=self.identity_interpret,
            output_shape=2
        )
        
        return qnn

    def grover_diffusion(self):
        n_qubits = 2
        diffusion_qr = QuantumRegister(n_qubits)
        diffusion_qc = QuantumCircuit(diffusion_qr, name='diffusion')
        
        diffusion_qc.h(range(0, 2))
        diffusion_qc.x(range(0, 2))
        diffusion_qc.h(1)
        diffusion_qc.cx(0, 1)
        diffusion_qc.h(1)
        diffusion_qc.x(range(0, 2))
        diffusion_qc.h(range(0, 2))
        
        # convert to a gate
        inst_diff = diffusion_qc.to_instruction()
        
        return inst_diff
    
    def grover(self):
        n_qubits = 5
        grover_qr = QuantumRegister(n_qubits)
        grover_qc = QuantumCircuit(grover_qr, name='Glover Gate')
        
        grover_qc.h(range(2, 4))
        grover_qc.x(4)
        grover_qc.h(4)
        grover_qc.cx(1, 3)
        grover_qc.cx(0, 2)
        
        grover_qc.barrier()
        grover_qc.x(range(2, 4))
        grover_qc.ccx(2, 3, 4)
        grover_qc.x(range(2, 4))
        grover_qc.barrier()
        
        grover_qc.cx(0, 2)
        grover_qc.cx(1, 3)
        
        grover_qc.barrier()
        
        grover_qc.append(self.grover_diffusion(), [grover_qr[2], grover_qr[3]])
        
        # print(grover_qc)
        
        inst_grover = grover_qc.to_instruction()
        
        return inst_grover

    # 分類回転ゲート
    def classify_rot_gate(self, n_qubits, state_phase):
    
        rot_qr = QuantumRegister(n_qubits)
        rot_qc = QuantumCircuit(rot_qr, name="Classify Rotation Gate")
        
        # 位相推定ゲートで角度を決める
        rot_qc.ry(state_phase, rot_qr)
        
        inst_rot = rot_qc.to_instruction()
        
        return inst_rot

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
            
        # label_para_qc.cx(0, 1)
        
        inst_qc = label_para_qc.to_instruction()
        
        return inst_qc

    # パラメータ化量子回路
    def parametarized_qcl(self):
        n_qubits = 4
        
        para_ent_qr = QuantumRegister(n_qubits)
        para_ent_qc = QuantumCircuit(para_ent_qr, name="Parametarized Gate")

        for id_qubit in range(n_qubits-1):
            
            self.add_paramed_ent_gate(para_ent_qc, id_qubit, 0, 1)
            
            para_ent_qc.barrier()
        
        inst_paramed_gate = para_ent_qc.to_instruction()
        
        return inst_paramed_gate
    
    # 最大エンタングルメント状態
    def max_entanglement_gate(self):
        n_qubits = 4
        
        max_ent_qr = QuantumRegister(n_qubits)
        max_ent_qc = QuantumCircuit(max_ent_qr, name="Max Entanglement Gate")
        
        max_ent_qc.h(0)
        
        for i in range(n_qubits-1):
            max_ent_qc.cx(i, i+1)
        inst_max_ent = max_ent_qc.to_instruction()
        return inst_max_ent
    
    # コスト関数
    def cost_func(self, weights, is_correct: bool):

        # 測定
        qnn = self.qcl_pred(self.x_train, self.y_train)
        probabilities = qnn.forward(input_data=None, weights=weights)        
        
        # 0 <= np.sum(probabilities[:, 1]) <= 0.5
        
        # コスト関数
        if is_correct:
            LOSS = (1 - np.sum(probabilities[:, 1]) - 0.5) * 2
        else:
            LOSS = (np.sum(probabilities[:, 1])) * 2

        return LOSS
    
    # 勾配計算
    def calc_gradient(self, params, is_correct):
        grad = np.zeros_like(params)
        for i in range(len(params)):
            shifted = params.copy()
            shifted[i] += np.pi/2
            forward = self.cost_func(shifted, is_correct)
            
            shifted[i] -= np.pi
            backward = self.cost_func(shifted, is_correct)
            
            grad[i] = 0.5 * (forward - backward)
        
        return np.round(grad, 10)
    
    # パラメータ更新
    def update_weights(self, weights, is_correct):
        lr = 1E-2
        
        grad = self.calc_gradient(weights, is_correct)
        # print("grad", grad)
        updated_weights = weights - lr * grad
        
        return updated_weights
    
    def predict(self, X_test, optimed_weight, is_correct: bool):
        innerproducts = np.array([])
        
        for y in range(NUM_CLASS):
            qnn = self.qcl_pred(X=X_test, y=y)
            probabilities = qnn.forward(input_data=None, weights=optimed_weight)
            
            innerproducts = np.append(innerproducts, np.sum(probabilities[:, 1]))
        
        print("innerproducts", innerproducts)
        
        return np.argmax(innerproducts) + 1

    def accuracy(self, X_test, y_test, optimed_weight, is_correct_test: bool):
        pred_dict = {}
        y_pred = np.array([])
        count = 0
        for x, y, is_correct in zip(X_test, y_test, is_correct_test):
            predicted = self.predict(X_test=x, optimed_weight=optimed_weight, is_correct=is_correct)
            y_pred = np.append(y_pred, predicted)
            pred_dict['ans{0} is {1}'.format(count, y)] = "pred{0} is {1}".format(count, predicted)
            
            count += 1
        
        print('y_pred', y_pred)
        # print('y_pred.shape', y_pred.shape)
        # print('type(y_pred)', type(y_pred))
        print('y_test', y_test)
        # print('y_test.shape', y_test.shape)
        # print('type(y_test)', type(y_test))
        
        # print("pred_dict :", pred_dict)
        print("accuracy_score :", accuracy_score(list(y_test), list(y_pred)))
        
        accuracy = accuracy_score(list(y_test), list(y_pred))
        
        graph_accuracy(accuracy, title="Accuracy value against iteration")
        
        return accuracy


def graph_loss(loss, y, title, is_correct):
    fig1, ax1 = plt.subplots()
    
    y_list.append(y)
    
    ax1.set_title(title+' ({})'.format(is_correct))
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("LOSS value")
    
    if is_correct:
        loss_func_vals_true.append(loss)
        loss_point_list = [y_list, loss_func_vals_true]
        print('loss_point_list', loss_point_list)
        ax1.plot(range(len(loss_func_vals_true)), loss_func_vals_true)
        
    else:
        loss_func_vals_false.append(loss)
        loss_point_list = [y_list, loss_func_vals_false]
        ax1.plot(range(len(loss_func_vals_false)), loss_func_vals_false)

    cnt_1 = 0
    cnt_2 = 0
    cnt_3 = 0
    cnt_4 = 0
    
    for (i, point), y in zip(enumerate(loss_point_list[-1]), y_list):
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
    
    if is_correct:
        plt.savefig(FIG_NAME_LOSS_TRUE)
    else:
        plt.savefig(FIG_NAME_LOSS_FALSE)
        
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


if __name__ == "__main__":
    
    for data_seed in [1, 10, 20, 30, 40, 50]:
    
        # simlator
        simulator = AerSimulator(device='GPU')
        # num shots
        nshots = 100000
        
        # seed value for params
        algorithm_globals.random_seed = PARAMS_SEED
        
        loss_func_vals_true = []
        loss_func_vals_false = []
        accuracy_vals = []
        
        loss_point = []
        y_list = []
        
        FOLDER_PATH = 'fig_v4/data_seed_{}/'.format(data_seed)
        FIG_NAME_LOSS_TRUE = FOLDER_PATH + 'graph_loss_true_seed.jpeg'
        FIG_NAME_LOSS_FALSE = FOLDER_PATH + 'graph_loss_false_seed.jpeg'
        FIG_NAME_ACCURACY = FOLDER_PATH + 'graph_accuracy_seed.jpeg'
        
        filepath_loss_true = Path(FIG_NAME_LOSS_TRUE)
        filepath_loss_false = Path(FIG_NAME_LOSS_FALSE)
        filepath_accuracy = Path(FIG_NAME_ACCURACY)
        filepath_loss_true.parent.mkdir(parents=True, exist_ok=True)
        filepath_loss_false.parent.mkdir(parents=True, exist_ok=True)
        filepath_accuracy.parent.mkdir(parents=True, exist_ok=True)
        
        
        if NUM_CLASS <= 4:
            
            df_dict = {}
            df = datasets(NUM_CLASS, NUM_FEATCHERS, DATA_SIZE)
            y = df["target"].values
            X = df.drop('target', axis=1).values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state = data_seed
            )
            
            is_correct_train = X_train[:, -1]
            is_correct_test = X_test[:, -1]
            
            X_train = X_train[:, :-1]
            X_test = X_test[:, :-1]
            
            y_list = list(set(y))
            
            initial_weights = 0.1 * (2 * algorithm_globals.random.random(N_PARAMS) - 1)
            optimized_weight = initial_weights
            
            # 学習
            for epoch in range(N_EPOCH):
                print("epoch : ", epoch+1)
                print("--------------------------------------------------------------------------------")
                
                for x, y, is_correct in zip(X_train, y_train, is_correct_train):
                    print("x", x)
                    print('y', y)
                    print('is_correct', is_correct)
                    
                    qc = SwapQNN(nqubits=N_QUBITS, simulator=simulator, nshots=nshots, X_train=x, y_train=y-1)
                    
                    # 最適化
                    optimized_weight = qc.update_weights(optimized_weight, is_correct=is_correct)
                    
                    graph_loss(qc.cost_func(optimized_weight, is_correct), y, title="Objective function value against iteration", is_correct=is_correct)
                    
                    # 推論
                    qc.accuracy(X_test, y_test, optimized_weight, is_correct_test)
        
        
        
    for params_seed in [1, 10, 20, 30, 40, 50]:
    
        # simlator
        simulator = AerSimulator(device='GPU')
        # num shots
        nshots = 100000
        
        # seed value for params
        algorithm_globals.random_seed = params_seed
        
        loss_func_vals_true = []
        loss_func_vals_false = []
        accuracy_vals = []
        
        loss_point = []
        y_list = []
        
        FOLDER_PATH = 'fig_v4/params_seed_{}/'.format(params_seed)
        FIG_NAME_LOSS_TRUE = FOLDER_PATH + 'graph_loss_true_seed.jpeg'
        FIG_NAME_LOSS_FALSE = FOLDER_PATH + 'graph_loss_false_seed.jpeg'
        FIG_NAME_ACCURACY = FOLDER_PATH + 'graph_accuracy_seed.jpeg'
        
        filepath_loss_true = Path(FIG_NAME_LOSS_TRUE)
        filepath_loss_false = Path(FIG_NAME_LOSS_FALSE)
        filepath_accuracy = Path(FIG_NAME_ACCURACY)
        filepath_loss_true.parent.mkdir(parents=True, exist_ok=True)
        filepath_loss_false.parent.mkdir(parents=True, exist_ok=True)
        filepath_accuracy.parent.mkdir(parents=True, exist_ok=True)
        
        
        if NUM_CLASS <= 4:
            
            df_dict = {}
            df = datasets(NUM_CLASS, NUM_FEATCHERS, DATA_SIZE)
            y = df["target"].values
            X = df.drop('target', axis=1).values
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state = DATA_SEED
            )
            
            is_correct_train = X_train[:, -1]
            is_correct_test = X_test[:, -1]
            
            X_train = X_train[:, :-1]
            X_test = X_test[:, :-1]
            
            y_list = list(set(y))
            
            initial_weights = 0.1 * (2 * algorithm_globals.random.random(N_PARAMS) - 1)
            optimized_weight = initial_weights
            
            # 学習
            for epoch in range(N_EPOCH):
                print("epoch : ", epoch+1)
                print("--------------------------------------------------------------------------------")
                
                for x, y, is_correct in zip(X_train, y_train, is_correct_train):
                    print("x", x)
                    print('y', y)
                    print('is_correct', is_correct)
                    
                    qc = SwapQNN(nqubits=N_QUBITS, simulator=simulator, nshots=nshots, X_train=x, y_train=y-1)
                    
                    # 最適化
                    optimized_weight = qc.update_weights(optimized_weight, is_correct=is_correct)
                    
                    graph_loss(qc.cost_func(optimized_weight, is_correct), y, title="Objective function value against iteration", is_correct=is_correct)
                    
                    # 推論
                    qc.accuracy(X_test, y_test, optimized_weight, is_correct_test)


