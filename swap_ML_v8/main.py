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
from qiskit.algorithms.optimizers import COBYLA

NUM_CLASS = 4
NUM_FEATCHERS = 4
DATA_SIZE = 200
N_PARAMS = 12*3 + 12*4
MAX_ITER = 500
N_EPOCH = 10

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
        # print(name)
        df_dict[name] = group.reset_index(drop=True)
        df_dict[name] = df_dict[name].iloc[range(int(data_size / 4)), :]
    
    df = pd.concat([df_dict[n] for n in df['target'].unique()], ignore_index=True)
    # shaffle datasets
    df = df.sample(frac=1, ignore_index=True)
    
    return df


###############################################################

def optimize(cost_func, initial_weights):
    
    opt = COBYLA(maxiter=MAX_ITER)
    opt_result = opt.minimize(cost_func, initial_weights)
    theta_opt = opt_result.x
    return theta_opt

def state_phase(y):
    qpe = QPE()
    state_phase = qpe.predict_qp(y)
    return state_phase * 2 * np.pi

# 位相推定アルゴリズム
class QPE:
    def __init__(self) -> None:
        # 第1レジスタのビット数
        self.N_ENCODE = 10
        # 第2レジスタのビット数
        self.N_EIGEN_STATE = 2
        
        self.N_QUBITS = self.N_ENCODE + self.N_EIGEN_STATE
        
        self.simulator = AerSimulator(device='GPU')
        self.nshots = 100000
    
    # 逆フーリエ変換
    def inverse_qft(self, n):
        qft = QFT(n)
        inv_qft = qft.inverse()
        
        return inv_qft
    
    def qpe(self, y):
        
        qr = QuantumRegister(self.N_QUBITS)
        cr = ClassicalRegister(self.N_ENCODE)
        qc = QuantumCircuit(qr, cr)
        
        # 位相推定したい状態
        y_bin = format(y, '02b')
        for i, y_i in enumerate(y_bin):
            if y_i == "1":
                qc.x(self.N_ENCODE+i)
        
        qc.barrier()
        
        for i in range(self.N_ENCODE):
            qc.h(i)
        
        theta = Parameter('θ')
        r = 1
        for c in range(self.N_ENCODE):
            for i in range(r):
                qc.cp(theta, control_qubit=c, target_qubit=self.N_QUBITS-1)
                qc.cp(theta, control_qubit=c, target_qubit=self.N_QUBITS-2)
            r *= 2
        
        # print(qc)
        
        qc.append(self.inverse_qft(self.N_QUBITS), qc.qubits[:self.N_QUBITS])
        
        qc.barrier()
        
        for i in range(self.N_ENCODE):
            qc.measure(i, i)
        
        # print(qc)
                
        return qc, theta
    
    def predict_qp(self, y):
        qc, theta = self.qpe(y)
        
        # 正解値 (θ=2*pi*φ)
        np.random.seed(0)
        phase = np.random.rand()
        # 回路にパラメータを設定する
        qc_para = qc.bind_parameters({theta: phase})
        
        compiled_circuit = transpile(qc_para, self.simulator)
        qobj = assemble(compiled_circuit, shots=self.nshots)
        results = self.simulator.run(qobj, shots=self.nshots).result()

        answer = results.get_counts()
        
        # 測定された位相の算出
        values = list(answer.values())
        keys = list(answer.keys())
        idx = np.argmax(list(values))
        # keys[idx]が2進数としてintに変換
        ans = int(keys[idx], 2)
        phase_estimated = ans / (2**self.N_ENCODE)
        
        return phase_estimated        
        

# メインの量子回路
class SwapQNN:
    def __init__(self, nqubits, simulator, nshots, state_phase, X_train: list, y_train: int) -> None:
        self.nqubits = nqubits
        self.simulator = simulator
        self.nshots = nshots
        self.n_params = N_PARAMS
        
        self.x_train = X_train
        self.y_train = y_train
        
        self.state_phase = state_phase
        
        self.weights = ParameterVector('weight', self.n_params)
        
    def U_in(self, X, y):
        qr = QuantumRegister(self.nqubits, 'qr')
        cr = ClassicalRegister(1, 'cr')
        qc = QuantumCircuit(qr, cr)
        
        # len(y_bin) == 2
        y_bin = format(y, '02b')
        
        # encode for label
        for i, y in enumerate(y_bin):
            if y == "1":
                qc.x(i)
                qc.x(i+7)
        
        # encode for input data
        print("X", X)
        for x in X:
            angle_y = np.arcsin(x)
            angle_z = np.arccos(x**2)
            
            for i in range(5, 5+2):
                qc.ry(angle_y, i)
                qc.rz(angle_z, i)
                
                qc.ry(angle_y, i+4)
                qc.rz(angle_z, i+4)
        
        return qc, qr
    
    # メインの量子回路
    def U_out(self, qc, qr):
        
        qc.append(self.grover(), [qr[i] for i in range(5)])
        qc.append(self.classify_rot_gate(n_qubits=9, state_phase=self.state_phase), [qr[i] for i in range(2, self.nqubits-1)])
        qc.append(self.max_entanglement_gate(), [qr[i] for i in range(7, self.nqubits-1)])
        qc.append(self.parametarized_qcl(), [qr[i] for i in range(2, 7)])
        
        qc.h(self.nqubits-1)
        
        # swap test
        for id in range(4):
            if id>=2:
                qc.cswap(self.nqubits-1, self.nqubits-2-id, self.nqubits-7-id)
            else:
                qc.cswap(self.nqubits-1, self.nqubits-2-id, self.nqubits-6-id)
        
        qc.h(self.nqubits-1)
        
        # qc.measure(self.nqubits-1, 0)
        
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

    def paramed_ent_gate(self, qc, id_qubit, skip_id1, skip_id2):
        qc.u(self.weights[id_qubit*12], self.weights[id_qubit*12+1], self.weights[id_qubit*12+2], id_qubit + skip_id1)
        qc.u(self.weights[id_qubit*12+3], self.weights[id_qubit*12+4], self.weights[id_qubit*12+5], id_qubit + skip_id2)
        qc.cx(id_qubit + skip_id1, id_qubit + skip_id2)
        qc.u(self.weights[id_qubit*12+6], self.weights[id_qubit*12+7], self.weights[id_qubit*12+8], id_qubit + skip_id1)
        qc.u(self.weights[id_qubit*12+9], self.weights[id_qubit*12+10], self.weights[id_qubit*12+11], id_qubit + skip_id2)

    def paramed_gate(self, qc, id_qubit, skip_id):
        qc.u(self.weights[id_qubit*12 + 12*3], self.weights[id_qubit*12+1 + 12*3], self.weights[id_qubit*12+2 + 12*3], id_qubit + skip_id)
        qc.u(self.weights[id_qubit*12+3 + 12*3], self.weights[id_qubit*12+4 + 12*3], self.weights[id_qubit*12+5 + 12*3], id_qubit + skip_id)
        qc.u(self.weights[id_qubit*12+6 + 12*3], self.weights[id_qubit*12+7 + 12*3], self.weights[id_qubit*12+8 + 12*3], id_qubit + skip_id)
        qc.u(self.weights[id_qubit*12+9 + 12*3], self.weights[id_qubit*12+10 + 12*3], self.weights[id_qubit*12+11 + 12*3], id_qubit + skip_id)


    # パラメータ化量子回路
    def parametarized_qcl(self):
        n_qubits = 5
        
        para_ent_qr = QuantumRegister(n_qubits)
        para_ent_qc = QuantumCircuit(para_ent_qr, name="Parametarized Gate")

        for id_qubit in range(n_qubits-2):
            
            if id_qubit == 1:
                self.paramed_ent_gate(para_ent_qc, id_qubit, 0, 2)
            elif id_qubit == 2:
                self.paramed_ent_gate(para_ent_qc, id_qubit, 1, 2)      
            else:
                self.paramed_ent_gate(para_ent_qc, id_qubit, 0, 1)
                
            para_ent_qc.barrier()
        
        for id_qubit in range(n_qubits-1):
            if id_qubit < 2:
                self.paramed_gate(para_ent_qc, id_qubit, 0)
            else:
                self.paramed_gate(para_ent_qc, id_qubit, 1)

        # print(para_ent_qc)
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
    def cost_func(self, weights):

        # 測定
        qnn = self.qcl_pred(self.x_train, self.y_train)
        probabilities = qnn.forward(input_data=None, weights=weights)
        
        # コスト関数
        LOSS = np.sum(probabilities[:, 1])
        print("LOSS", LOSS)
        return LOSS

    def initial_weights(self):
        initial_weights = 0.1 * (2 * algorithm_globals.random.random(self.n_params) - 1)
        return initial_weights
    
    # 最適化計算
    def minimization(self, initial_weights: list):
        theta_opt = optimize(self.cost_func, initial_weights)
        return theta_opt


class SwapQNN_pred():
    def __init__(self, simulator, n_qubits) -> None:
        self.simulator = simulator
        self.n_qubits = n_qubits

    def U_out_pred(self, qc, theta_opt):
        
        for id_qubit in range(self.n_qubits-1):
            qc.u(theta_opt[id_qubit*12], theta_opt[id_qubit*12+1], theta_opt[id_qubit*12+2], id_qubit)
            qc.u(theta_opt[id_qubit*12+3], theta_opt[id_qubit*12+4], theta_opt[id_qubit*12+5], id_qubit+1)
            qc.cx(id_qubit, id_qubit+1)
            qc.u(theta_opt[id_qubit*12+6], theta_opt[id_qubit*12+7], theta_opt[id_qubit*12+8], id_qubit)
            qc.u(theta_opt[id_qubit*12+9], theta_opt[id_qubit*12+10], theta_opt[id_qubit*12+11], id_qubit+1)
            
            qc.barrier()
        
        for id_qubit in range(self.n_qubits):
                qc.u(theta_opt[id_qubit*6 + 12*3], theta_opt[id_qubit*6+1 + 12*3], theta_opt[id_qubit*6+2 + 12*3], id_qubit)
                qc.u(theta_opt[id_qubit*6+3 + 12*3], theta_opt[id_qubit*6+4 + 12*3], theta_opt[id_qubit*6+5 + 12*3], id_qubit)
                qc.u(theta_opt[id_qubit*6+6 + 12*3], theta_opt[id_qubit*6+7 + 12*3], theta_opt[id_qubit*6+8 + 12*3], id_qubit)
                qc.u(theta_opt[id_qubit*6+9 + 12*3], theta_opt[id_qubit*6+10 + 12*3], theta_opt[id_qubit*6+11 + 12*3], id_qubit)
                
        return qc
        
    def U_in_pred(self, X_test, y):
        
        qr = QuantumRegister(self.n_qubits, 'qr')
        cr = ClassicalRegister(1, 'cr')
        qc = QuantumCircuit(qr, cr)
        
        y_bin = format(y, '02b')
        # encode for label
        for i, y in enumerate(y_bin):
            if y == '1':
                qc.x(i)
        
        # encode for input data
        for x in X_test:
            angle_y = np.arcsin(x)
            angle_z = np.arccos(x**2)
            
            for i in range(2, 4):
                qc.ry(angle_y, i)
                qc.rz(angle_z, i)
        
        return qc, qr
    
    # 分類回転ゲート
    def classify_rot_gate(self, state_phase):
    
        rot_qr = QuantumRegister(self.n_qubits)
        rot_qc = QuantumCircuit(rot_qr, name="Classify Rotation Gate")
        
        # 位相推定ゲートで角度を決める
        rot_qc.ry(state_phase, rot_qr)
        inst_rot = rot_qc.to_instruction()
        
        return inst_rot
        
    def predict(self, X_test, theta_opt):
        ent_entropies = np.array([])
        for y in range(NUM_CLASS):
            phase = state_phase(y)
            
            qc, qr = self.U_in_pred(X_test, y)
            qc.append(self.classify_rot_gate(state_phase=phase), [qr[i] for i in range(self.n_qubits)])
            qc = self.U_out_pred(qc, theta_opt)
            # print(qc)
            qc = transpile(qc, self.simulator)
            qc.save_statevector()
            
            job = self.simulator.run(qc)
            statevector = job.result().get_statevector(qc)
            
            reduced_state = partial_trace(statevector, [2, 3])
            ent_entropies = np.append(ent_entropies, entropy(reduced_state))
        
        # print("entanglement entropy :", ent_entropies)
        
        return np.argmax(ent_entropies) + 1

def accuracy(X_test, y_test, theta_opt):
    pred_dict = {}
    y_pred = []
    count = 0
    for x, y in zip(X_test, y_test):
        qc_pred = SwapQNN_pred(simulator=simulator, n_qubits=4)
        predicted = qc_pred.predict(X_test=x, theta_opt=theta_opt)
        y_pred.append(predicted)
        pred_dict['ans{0} : {1}'.format(count, y)] = "pred{0} : {1}".format(count, predicted)
        
        count += 1
        
    print("pred_dict :", pred_dict)
    print("accuracy_score :", accuracy_score(y_test, y_pred))
    
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy
    

def save_file_at_dir(dir_path, filename, dataframe):
    os.makedirs(dir_path, exist_ok=True)
    dataframe.to_csv(dir_path+filename)

if __name__ == "__main__":
    # num qubit
    nqubits = 12
    # simlator
    simulator = AerSimulator(device='GPU')
    # num shots
    nshots = 100000
    
    df_dict = {}
    df = datasets(NUM_CLASS, NUM_FEATCHERS, DATA_SIZE)
    y = df["target"].values
    X = df.drop('target', axis=1).values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    
    # 学習
    for epoch in range(N_EPOCH):
        print("epoch : ", epoch+1)
        count = 0
        # df_theta_opt = pd.DataFrame(data=theta_opt.reshape(1, -1), columns=["{0}th_theta".format(i+1) for i in range(N_PARAMS)])
        # df_accuracy = pd.DataFrame()
        
        for x, y in zip(X_train, y_train):      
            count += 1  
            phase = state_phase(y-1)
            print("x", x)
            print('y', y)
            print("phase", phase)
            qc = SwapQNN(nqubits=nqubits, simulator=simulator, nshots=nshots, state_phase=phase, X_train=x, y_train=y-1)
            initial_theta = qc.initial_weights()

            # 最適化
            theta_opt = qc.minimization(initial_theta)
            # row, col = df_theta_opt.shape
            # df_theta_opt.loc[row] = theta_opt
            
            # 推論
            accuracy_ = accuracy(X_test, y_test, theta_opt)
            # accuracy_series = pd.Series([accuracy_])
            # df_accuracy = pd.concat([df_accuracy, accuracy_series])
            # df_accuracy = df_accuracy.reset_index(drop=True)            
        
        # ファイル出力
        # save_file_at_dir('weights/', '{0}th_epoch.csv'.format(epoch+1), df_theta_opt)
        # save_file_at_dir('accuracy/', '{0}th_epoch.csv'.format(epoch+1), df_accuracy)
        
        
    
    # # 推論
    # pred_dict = {}
    # y_pred = []
    # count = 0
    # for x, y in zip(X_test, y_test):
    #     phase_test = state_phase(y)
    #     qc_pred = SwapQNN_pred(simulator=simulator, n_qubits=4)
    #     predicted = qc_pred.predict(X_test=x, theta_opt=theta_opt)
    #     y_pred.append(predicted)
    #     pred_dict['ans{0} : {1}'.format(count, y)] = "pred{0} : {1}".format(count, predicted)
        
    #     count += 1
        
    # print("pred_dict :", pred_dict)
    # print("accuracy_score :", accuracy_score(y_test, y_pred))