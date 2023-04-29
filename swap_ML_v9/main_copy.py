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
from qiskit.circuit import ParameterVector, Parameter
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizers import COBYLA, ADAM, NELDER_MEAD

NUM_CLASS = 4
NUM_FEATCHERS = 4
DATA_SIZE = 40
N_PARAMS = 12*3 + 12*4
MAX_ITER = 1000
XATOL = 0.0001
N_EPOCH = 10
N_QUBITS = 12

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
        try:
            y_bin = format(y, '02b')
            for i, y_i in enumerate(y_bin):
                if y_i == "1":
                    qc.x(self.N_ENCODE+i)
        except:
            pass
        
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
        
        qc.append(self.inverse_qft(self.N_QUBITS), qc.qubits[:self.N_QUBITS])
        
        qc.barrier()
        
        for i in range(self.N_ENCODE):
            qc.measure(i, i)
                
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
    def __init__(self, nqubits, simulator):
        self.nqubits_train = nqubits
        self.simulator = simulator
        self.n_params = N_PARAMS
        
        self.x = ParameterVector('x', NUM_CLASS)
        self.weights = ParameterVector('weight', self.n_params)
        
    def U_in(self, y):
        qr = QuantumRegister(self.nqubits_train, 'qr')
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
        for x in [self.x[i] for i in range(NUM_CLASS)]:
            angle_y = np.arcsin(x)
            angle_z = np.arccos(x*x)
            
            print('angle_y', angle_y)
            
            for i in range(5, 5+2):
                qc.ry(angle_y, i)
                qc.rz(angle_z, i)
                
                qc.ry(angle_y, i+4)
                qc.rz(angle_z, i+4)
        
        return qc, qr
    
    # メインの量子回路
    def U_out(self, qc, qr):
        
        qc.append(self.grover(), [qr[i] for i in range(5)])
        # qc.append(self.classify_rot_gate(n_qubits=9, state_phase=self.state_phase), [qr[i] for i in range(2, self.nqubits_train-1)])
        qc.append(self.max_entanglement_gate(), [qr[i] for i in range(7, self.nqubits_train-1)])
        qc.append(self.parametarized_qcl(), [qr[i] for i in range(2, 7)])
        
        qc.h(self.nqubits_train-1)
        
        # swap test
        for id in range(4):
            if id>=2:
                qc.cswap(self.nqubits_train-1, self.nqubits_train-2-id, self.nqubits_train-7-id)
            else:
                qc.cswap(self.nqubits_train-1, self.nqubits_train-2-id, self.nqubits_train-6-id)
        
        qc.h(self.nqubits_train-1)
        
        # qc.measure(self.nqubits_train-1, 0)
        
        return qc

    def identity_interpret(self, x):
        return x
    
    def qcl_pred(self, y):
        qc, qr = self.U_in(y)
        qc = self.U_out(qc, qr)
        qc.measure(self.nqubits_train-1, 0)
        
        return qc
        

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


    # パラメータ化量子回路
    def parametarized_qcl(self):
        n_qubits = 5
        
        para_ent_qr = QuantumRegister(n_qubits)
        para_ent_qc = QuantumCircuit(para_ent_qr, name="Parametarized Gate")

        for id_qubit in range(n_qubits-2):
            
            if id_qubit == 1:
                self.add_paramed_ent_gate(para_ent_qc, id_qubit, 0, 2)
            elif id_qubit == 2:
                self.add_paramed_ent_gate(para_ent_qc, id_qubit, 1, 2)      
            else:
                self.add_paramed_ent_gate(para_ent_qc, id_qubit, 0, 1)
                
            para_ent_qc.barrier()
        
        for id_qubit in range(n_qubits-1):
            if id_qubit < 2:
                self.add_paramed_gate(para_ent_qc, id_qubit, 0)
            else:
                self.add_paramed_gate(para_ent_qc, id_qubit, 1)

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
    

# 評価用クラス
class Evaluate:
    def __init__(self, qnn, input_data):
        self.qnn = qnn
        self.input_data = input_data
        self.counts = 0
  
    # 最適化計算
    def optimize(self, initial_weights):    
        opt = COBYLA(maxiter=MAX_ITER)
        opt_result = opt.minimize(self.cost_func, initial_weights)
        theta_opt = opt_result.x
        return theta_opt
    
    def cost_func(self, weights):
        self.counts += 1
        # 測定
        print("self.input_data",self.input_data)
        print("weights", weights)
        print(len(weights))
        print(self.counts)
        probabilities = self.qnn.forward(input_data=self.input_data, weights=weights)
        # コスト関数
        print("probabilities", probabilities)
        LOSS = np.sum(probabilities[:, 1])
        print("LOSS", LOSS)
        
        return LOSS
    
    # コスト関数の最小化
    def minimize(self, initial_weights):
        optimized_weights = optimize(self.cost_func, initial_weights)
        return optimized_weights
    
    # 推論
    def predict(self, input_data, optimized_weights):
        innerproducts = np.array([])
        for y in range(NUM_CLASS):
            probabilities = self.qnn.forward(input_data=input_data, weights=optimized_weights)
            LOSS = np.sum(probabilities[:, 1])
            innerproducts = np.append(innerproducts, 1-LOSS)
        
        return np.argmax(innerproducts) + 1
    
    
    # 精度
    def accuracy(self, X_test, y_test, predicted):
        pred_dict = {}
        y_pred = np.array([])
        count = 0
        for x, y in zip(X_test, y_test):
            y_pred = np.append(y_pred, predicted)
            pred_dict['ans{0} : {1}'.format(count, y)] = "pred{0} : {1}".format(count, predicted)
            
            count += 1
        
        print('y_pred', y_pred)
        
        print("pred_dict :", pred_dict)
        print("accuracy_score :", accuracy_score(y_test, y_pred))
        
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    

def save_file_at_dir(dir_path, filename, dataframe):
    os.makedirs(dir_path, exist_ok=True)
    dataframe.to_csv(dir_path+filename)

def identity_interpret(x):
    return x

if __name__ == "__main__":
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
    
    # 全体の量子回路
    swapqnn = SwapQNN(nqubits=N_QUBITS, simulator=simulator)
    
    initial_weights = 0.1 * (2 * algorithm_globals.random.random(N_PARAMS) - 1)
    optimized_weight = initial_weights
    
    # 学習
    for epoch in range(N_EPOCH):
        print("epoch : ", epoch+1)
        print('----------------------------------------------------------------------')
        
        # ここでxとyをくっつけてシャッフル
        # モデルを先に定義しておいて後で、xとyを入力
        # SamplerQNNモデルを外で定義する
        
        for x, y in zip(X_train, y_train):
            print("x", x)
            print('y', y)
            
            u_in, _ = swapqnn.U_in(y)
            print(u_in.parameters)
            
            # input_data = np.append(x, y)
            input_data = x
            
            print("input_data", input_data)
            
            qc = swapqnn.qcl_pred(y)
            
            print(qc)
            
            qnn = SamplerQNN(
                circuit=qc,
                input_params=u_in.parameters,
                weight_params=swapqnn.weights,
                interpret=identity_interpret,
                output_shape=2,
            )
            
            evaluate = Evaluate(qnn, input_data)
            
            # 最適化
            optimized_weight = evaluate.minimize(optimized_weight)
            print('pass')
            
            # 推論
            accuracy_score = evaluate.accuracy(X_test, y_test, evaluate.predict(input_data=input_data, optimized_weights=optimized_weight))
            
            