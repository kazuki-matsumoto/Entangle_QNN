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
import torch

NUM_CLASS = 4
NUM_FEATCHERS = 4
DATA_SIZE = 100
N_PARAMS = 12*3
MAX_ITER = 500

OPTIM_STEPS = 100

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

    # データ数の指定
    df = df.iloc[range(data_size), :]    
    
    return df


###############################################################
# pytorchの最適化に置き換える!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def optimize(cost_func, theta_init, method='Nelder-Mead'):
    result = minimize(cost_func, theta_init, method=method, options={'maxiter':MAX_ITER})
    theta_opt = result.x
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
        for i, y in enumerate(y_bin):
            if y == "1":
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
        
        qc.append(self.inverse_qft(self.N_QUBITS), qc.qubits[:self.N_QUBITS])
        
        qc.barrier()
        
        for i in range(self.N_ENCODE):
            qc.measure(i, i)
        
        return qc, theta
    
    def predict_qp(self, y):
        qc, theta = self.qpe(y)
        
        # 正解値 (θ=2*pi*φ)
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
        idx = np.argmax(values)
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
        # self.state_phase = self.state_phase()
        
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
    def U_out(self, qc, qr, thetas):
        
        # print(thetas)
        
        qc.append(self.grover(), [qr[i] for i in range(5)])
        qc.append(self.classify_rot_gate(n_qubits=9, state_phase=self.state_phase), [qr[i] for i in range(2, self.nqubits-1)])
        qc.append(self.max_entanglement_gate(), [qr[i] for i in range(7, self.nqubits-1)])
        qc.append(self.parametarized_gate(thetas=thetas), [qr[i] for i in range(2, 7)])
        
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

    def qcl_pred(self, theta, X, y):
        qc, qr = self.U_in(X, y)
        self.qc = self.U_out(qc, qr, theta)
        self.qc.measure(self.nqubits-1, 0)
        counts = self.run_circuit(self.qc)
        
        if '1' in counts:
            b = counts['1']
        else:
            b = 0
        
        # 忠実度の計算
        inner_product = 1-(2/self.nshots)*b
        
        return inner_product

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

    # パラメータ化量子回路
    def parametarized_gate(self, thetas):
        n_qubits = 5
        
        para_ent_qr = QuantumRegister(n_qubits)
        para_ent_qc = QuantumCircuit(para_ent_qr, name="Parametarized Gate")

        for id_qubit in range(n_qubits-2):
            
            if id_qubit == 1:
                para_ent_qc.u(thetas[id_qubit*12], thetas[id_qubit*12+1], thetas[id_qubit*12+2], id_qubit)
                para_ent_qc.u(thetas[id_qubit*12+3], thetas[id_qubit*12+4], thetas[id_qubit*12+5], id_qubit+2)
                para_ent_qc.cx(id_qubit, id_qubit+2)
                para_ent_qc.u(thetas[id_qubit*12+6], thetas[id_qubit*12+7], thetas[id_qubit*12+8], id_qubit)
                para_ent_qc.u(thetas[id_qubit*12+9], thetas[id_qubit*12+10], thetas[id_qubit*12+11], id_qubit+2)
            
            elif id_qubit == 2:
                para_ent_qc.u(thetas[id_qubit*12], thetas[id_qubit*12+1], thetas[id_qubit*12+2], id_qubit+1)
                para_ent_qc.u(thetas[id_qubit*12+3], thetas[id_qubit*12+4], thetas[id_qubit*12+5], id_qubit+2)
                para_ent_qc.cx(id_qubit+1, id_qubit+2)
                para_ent_qc.u(thetas[id_qubit*12+6], thetas[id_qubit*12+7], thetas[id_qubit*12+8], id_qubit+1)
                para_ent_qc.u(thetas[id_qubit*12+9], thetas[id_qubit*12+10], thetas[id_qubit*12+11], id_qubit+2)
                        
            else:
                para_ent_qc.u(thetas[id_qubit*12], thetas[id_qubit*12+1], thetas[id_qubit*12+2], id_qubit)
                para_ent_qc.u(thetas[id_qubit*12+3], thetas[id_qubit*12+4], thetas[id_qubit*12+5], id_qubit+1)
                para_ent_qc.cx(id_qubit, id_qubit+1)
                para_ent_qc.u(thetas[id_qubit*12+6], thetas[id_qubit*12+7], thetas[id_qubit*12+8], id_qubit)
                para_ent_qc.u(thetas[id_qubit*12+9], thetas[id_qubit*12+10], thetas[id_qubit*12+11], id_qubit+1)
            
            para_ent_qc.barrier()
        
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

    # 回路の実行
    def run_circuit(self, qc):
        compiled_circuit = transpile(qc, self.simulator)
        job = self.simulator.run(compiled_circuit, shots=self.nshots)
        result = job.result()
        
        counts = result.get_counts(qc)
        
        return counts
    
    # コスト関数を計算
    def cost_func(self, thetas):
        
        # 測定（忠実度の計算）
        output = self.qcl_pred(thetas, self.x_train, self.y_train)
        print('output :', output)
        
        # コスト関数
        LOSS = (output - 1)**2
        print('LOSS :', LOSS)
        
        return LOSS

    def initial_theta(self):
        np.random.seed(0)
        theta = np.random.rand(self.n_params) * 2.0 * np.pi
        return theta
    
    # 最適化計算
    def minimization(self, initial_theta: list):
        theta_opt = optimize(self.cost_func, initial_theta)
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
            
            qc = transpile(qc, self.simulator)
            qc.save_statevector()
            
            job = self.simulator.run(qc)
            statevector = job.result().get_statevector(qc)
            
            reduced_state = partial_trace(statevector, [2, 3])
            ent_entropies = np.append(ent_entropies, entropy(reduced_state))
        
        print("entanglement entropy :", ent_entropies)
        
        return np.argmax(ent_entropies)

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
    
    print(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    # theta_opt = np.random.rand(N_PARAMS) * 2.0 * np.pi
    theta_opt = np.zeros(N_PARAMS)
    
    # 学習
    for x, y in zip(X_train, y_train):
        phase = state_phase(y)
        qc = SwapQNN(nqubits=nqubits, simulator=simulator, nshots=nshots, state_phase=phase, X_train=x, y_train=y)
        initial_theta = theta_opt

        theta_opt = qc.minimization(initial_theta)
    
    # 推論
    pred_dict = {}
    y_pred = []
    count = 0
    for x, y in zip(X_test, y_test):
        phase_test = state_phase(y)
        qc_pred = SwapQNN_pred(simulator=simulator, n_qubits=4)
        predicted = qc_pred.predict(X_test=x, theta_opt=theta_opt)
        y_pred.append(predicted)
        pred_dict['ans{0} : {1}'.format(count, y)] = "pred{0} : {1}".format(count, predicted)
        
        count += 1
        
    print("pred_dict :", pred_dict)
    print("accuracy_score :", accuracy_score(y_test, y_pred))