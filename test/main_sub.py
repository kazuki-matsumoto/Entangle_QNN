from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from qulacs import Observable, QuantumState, QuantumCircuit, ParametricQuantumCircuit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qulacs.gate import DenseMatrix
from scipy.optimize import minimize

def dataset():
    data = load_iris()
    print(data.target)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df = df.iloc[:,:2]
    df['target'] = data.target
 
    return df

#### メインクラス
class Main:
    def __init__(self, n_qubits) -> None:
        self.n_qubits = n_qubits
        self.U_out = self.U_out(n_qubits)
        
        self.observable = Observable(n_qubits)
        self.observable.add_operator(1, 'Z 2')
        
        
    # 分類タスクに合うように変更する
    # エンコード
    def U_in(self, x):
        U = QuantumState(self.n_qubits)
        
        angle_y = np.arcsin(x)
        angle_z = np.arccos(x**2)
        
        for i in range(self.n_qubits-3):
            U.add_RY_gate(i, angle_y)
            U.add_RZ_gate(i, angle_z)
        
        return U
    
    # CCXゲート
    def toffoli_gate(self):
        toffoli = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0]
        ]
        
        return DenseMatrix([2, 3, 4], toffoli)
    
    # パラメータ化されたゲートが含まれた回路
    def U_out(self, n_qubits=5):
        U = ParametricQuantumCircuit(n_qubits)
        
        # parametric gates
        for i in range(2):
            angle = 2*np.pi*np.random.rand()
            U.add_parametric_RZ_gate(i, angle)
            angle = 2*np.pi*np.random.rand()
            U.add_parametric_RY_gate(i, angle)
            angle = 2*np.pi*np.random.rand()
            U.add_parametric_RZ_gate(i, angle)
        
        U.add_H_gate(2)
        U.add_H_gate(3)
        U.add_X_gate(4)
        U.add_H_gate(4)
        
        U.add_CNOT_gate(1, 3)
        U.add_CNOT_gate(0, 2)
        
        U.add_X_gate(2)
        U.add_X_gate(3)
        
        # toffoli gate
        U.add_gate(self.toffoli_gate())
        
        U.add_X_gate(2)
        U.add_X_gate(3)
        
        U.add_CNOT_gate(0, 2)
        U.add_CNOT_gate(1, 3)
        
        # diffusion gate
        U.add_H_gate(2)
        U.add_H_gate(3)
        U.add_X_gate(2)
        U.add_X_gate(3)
        U.add_H_gate(3)
        U.add_CNOT_gate(2,3)
        U.add_H_gate(3)
        U.add_X_gate(2)
        U.add_X_gate(3)
        U.add_H_gate(2)
        U.add_H_gate(3)
        
        # parametric gates
        for i in range(2):
            angle = 2*np.pi*np.random.rand()
            U.add_parametric_RZ_gate(i+2, angle)
            angle = 2*np.pi*np.random.rand()
            U.add_parametric_RY_gate(i+2, angle)
            angle = 2*np.pi*np.random.rand()
            U.add_parametric_RZ_gate(i+2, angle)
        
        return U
    
    # パラメータthetaを更新する関数
    def set_U_out(self, theta):
        params_cnt = self.U_out.get_parameter_count()
        
        for i in range(params_cnt):
            self.U_out.set_parameter(i, theta[i])
        
    # 回路の予測結果を出力
    def predict(self, x):
        state = QuantumState(self.n_qubits)
        state.set_zero_state()
        
        # 入力状態の準備
        self.U_in(x).update_quantum_state(state)
        
        # 出力状態の計算
        self.U_out.update_quantum_state(state)
        
        # モデルの出力
        res = self.observable.get_expectation_value(state)
        
        return res
    
    # コスト関数の定義
    def cost(self, theta):
        self.set_U_out(theta)
        
        # 予測結果の出力
        y_pred = [self.predict(x) for x in X]
        

if __name__ == '__main__':
    main = Main(n_qubits=5)