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
from qiskit.quantum_info import partial_trace, entropy


NUM_CLASS = 4
NUM_FEATCHERS = 4
DATA_SIZE = 100
# N_PARAMS = 12*3 + 12*4
N_PARAMS = 12*3
MAX_ITER = 500
N_EPOCH = 10
N_QUBITS = 4
LEARNING_RATE = 0.001

OPTIM_STEPS = 100

# Set seed for random generators
algorithm_globals.random_seed = 42


###############################################################

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
        df_dict[name] = group.reset_index(drop=True)
        df_dict[name] = df_dict[name].iloc[range(int(data_size / 4)), :]
    
    df = pd.concat([df_dict[n] for n in df['target'].unique()], ignore_index=True)
    # shaffle datasets
    df = df.sample(frac=1, ignore_index=True)
    
    return df


###############################################################

def optimize(cost_func, initial_weights):
    
    opt = COBYLA(maxiter=MAX_ITER)
    # opt = ADAM(maxiter=MAX_ITER)
    opt_result = opt.minimize(cost_func, initial_weights)
    theta_opt = opt_result.x
    return theta_opt


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
        
        # encode for input data
        for x in X:
            angle_y = np.arcsin(x)
            angle_z = np.arccos(x**2)
            
            for i in range(2, 2+2):
                qc.ry(angle_y, i)
                qc.rz(angle_z, i)
        
        # print(qc)
        
        return qc, qr
    
    # メインの量子回路
    def U_out(self, qc, qr):
        
        qc.append(self.parametarized_qcl(), [qr[i] for i in range(0, 4)])
        
        return qc

    def identity_interpret(self, x):
        return x
    
    def qcl_pred(self, X, y):
        qc, qr = self.U_in(X, y)
        qc = self.U_out(qc, qr)
        
        return qc

    def add_paramed_ent_gate(self, qc, id_qubit, skip_id1, skip_id2):
        qc.u(self.weights[id_qubit*12], self.weights[id_qubit*12+1], self.weights[id_qubit*12+2], id_qubit + skip_id1)
        qc.u(self.weights[id_qubit*12+3], self.weights[id_qubit*12+4], self.weights[id_qubit*12+5], id_qubit + skip_id2)
        qc.cx(id_qubit + skip_id1, id_qubit + skip_id2)
        qc.u(self.weights[id_qubit*12+6], self.weights[id_qubit*12+7], self.weights[id_qubit*12+8], id_qubit + skip_id1)
        qc.u(self.weights[id_qubit*12+9], self.weights[id_qubit*12+10], self.weights[id_qubit*12+11], id_qubit + skip_id2)


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
    
    # コスト関数を計算
    def cost_func(self, weights, X, y):
        
        # 量子回路
        qc = self.qcl_pred(X, y)
        
        # 量子回路にパラメータを適用する
        weights_dct = dict(list(map(lambda x, y: [x, y], self.weights, weights)))
        qc.assign_parameters(weights_dct, inplace=True)
        
        # エンタングルメントエントロピーの計算
        qc = transpile(qc, self.simulator)
        qc.save_statevector()
        job = self.simulator.run(qc)
        statevector = job.result().get_statevector(qc)
        reduced_state = partial_trace(statevector, [0, 1])
        
        # コスト関数
        LOSS = 1 - entropy(reduced_state)
        return LOSS
    
    
    # 勾配計算
    def calc_gradient(self, params):
        grad = np.zeros_like(params)
        for i in range(len(params)):
            shifted = params.copy()
            shifted[i] += np.pi/2
            forward = self.cost_func(shifted, self.x_train, self.y_train)
            
            shifted[i] -= np.pi
            backward = self.cost_func(shifted, self.x_train, self.y_train)
            
            grad[i] = 0.5 * (forward - backward)

        return np.round(grad, 10)
    
    # パラメータ更新
    def update_weights(self, weights):
        grad = self.calc_gradient(weights)
        # print("grad", grad)
        updated_weights = weights - grad
        
        return updated_weights
        
    
    def predict(self, x_test, optimed_weight):
        entropies = np.array([])
        
        for y in range(NUM_CLASS):
            
            LOSS = self.cost_func(optimed_weight, X=x_test, y=y)
            # print("LOSS", LOSS)
            entropies = np.append(entropies, 1-LOSS)
         
        print("entropies", entropies)
         
        return np.argmax(entropies) + 1


    def accuracy(self, X_test, y_test, optimed_weight):
        pred_dict = {}
        y_pred = np.array([])
        count = 0
        
        for x, y in zip(X_test, y_test):
            predicted = self.predict(x_test=x, optimed_weight=optimed_weight)
            y_pred = np.append(y_pred, predicted)
            pred_dict['ans{0} is {1}'.format(count, y)] = "pred{0} is {1}".format(count, predicted)
            
            count += 1
        
        print('y_pred', y_pred)
        print("pred_dict :", pred_dict)
        print("accuracy_score :", accuracy_score(y_test, y_pred))
        
        accuracy = accuracy_score(y_test, y_pred)
        
        graph_accuracy(accuracy, title="Accuracy value against iteration")
        
        return accuracy


def graph_loss(loss, y, title):
    fig1, ax1 = plt.subplots()
    loss_func_vals.append(loss)
    y_list.append(y)
    loss_point_list = [y_list, loss_func_vals]
    
    ax1.set_title(title)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("LOSS value")
    ax1.plot(range(len(loss_func_vals)), loss_func_vals)
    
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

if __name__ == "__main__":
    # simlator
    simulator = AerSimulator(device='GPU')
    # num shots
    nshots = 100000
    
    loss_func_vals = []
    accuracy_vals = []
    
    loss_point = []
    y_list = []
    
    FIG_NAME_LOSS = 'fig_v1/graph_loss.jpeg'
    FIG_NAME_ACCURACY = 'fig_v1/graph_accuracy.jpeg'
    
    if NUM_CLASS <= 4:
        
        df_dict = {}
        df = datasets(NUM_CLASS, NUM_FEATCHERS, DATA_SIZE)
        y = df["target"].values
        X = df.drop('target', axis=1).values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=1
        )
        
        initial_weights = 0.1 * (2 * algorithm_globals.random.random(N_PARAMS) - 1)
        optimized_weight = initial_weights  
        
        # 学習
        for epoch in range(N_EPOCH):
            print("epoch : ", epoch+1)
            print("--------------------------------------------------------------------------------")
            
            for x, y in zip(X_train, y_train):
                print("x", x)
                print('y', y)
                
                qc = SwapQNN(nqubits=N_QUBITS, simulator=simulator, nshots=nshots, X_train=x, y_train=y-1)
                
                # 最適化
                # optimized_weight = qc.minimization(optimized_weight)
                optimized_weight = qc.update_weights(optimized_weight)
                print("optimized_weight", optimized_weight)
                # graph_loss(qc.cost_func(optimized_weight), y, title="Objective function value against iteration")
                
                # print("gradients",qc.calc_gradient(optimized_weight))
                
                # 推論
                qc.accuracy(X_test, y_test, optimized_weight)
                # qc_pred.accuracy(X_test, y_test, optimized_weight)
                
                
                
                
                
                
                
                
                