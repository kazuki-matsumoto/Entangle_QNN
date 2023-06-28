import numpy as np
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from ent_utils import (softmax, 
                       save_graph_loss, 
                       save_graph_grad)
from sklearn.metrics import accuracy_score

from qiskit import *
from qiskit.circuit import ParameterVector
from qiskit.utils import algorithm_globals
from qiskit.primitives import BackendEstimator
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer
import mthree

from qiskit.opflow import Z, StateFn, CircuitStateFn, PauliExpectation, CircuitSampler
from qiskit.utils.quantum_instance import QuantumInstance


class QclClassification:
    
    def __init__(
        self, 
        nqubit, 
        data_nqubits, 
        class_nqubits, 
        num_features, 
        c_depth, num_class, 
        fig_name_loss,
        fig_name_grad,
        num_epochs,
        batch_size,
        num_measure=9):
        
        """
        :param nqubit: qubitの数。必要とする出力の次元数よりも多い必要がある
        :param c_depth: circuitの深さ
        :param num_class: 分類の数
        :num_measure: 測定するqubitの数
        """
        
        self.nqubit = nqubit
        
        self.data_nqubits = data_nqubits
        self.class_nqubits = class_nqubits
        
        self.num_features = num_features
        
        self.c_depth = c_depth

        self.input_state_list = []  # |ψ_in>のリスト
        self.theta = []  # θのリスト

        self.output_gate = None  # U_out

        self.num_class = num_class  # 分類の数
        
        self.n_params = self.nqubit * self.c_depth # パラメータの数
        self.weights = ParameterVector('weight', self.n_params) # パラメータのベクトル
        
        self.fig_name_loss = fig_name_loss
        self.fig_name_grad = fig_name_grad
        
        backend = AerSimulator()
        self.estimator = BackendEstimator(backend)
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # オブザーバブルの準備
        self.obs = Z^Z^Z^Z^Z^Z^Z^Z^Z
        
        print("self.obs", self.obs)

    def U_in(self, x, y):
        
        # 単一の入力x, yをエンコードするゲートを作成する関数
        # xは入力特徴量(2次元)。
        # yはオンホットベクトル
        # xの要素は[-1, 1]の範囲内
        
        qr = QuantumRegister(self.nqubit, 'qr')
        qc = QuantumCircuit(qr)
        
        index_y = np.argmax(y[:-1])
        
        # encode for label (y)
        for i, y_i in enumerate(format(index_y, f'0{self.class_nqubits}b')):
            if y_i == "1":
                qc.x(i)
                qc.x(i + self.data_nqubits + self.class_nqubits)
        
        angle_y = np.arcsin(x.astype(np.float64))
        angle_z = np.arccos((x**2).astype(np.float64))
        
        # encode for input data (x)
        for j in range(self.num_features - self.data_nqubits):
            for i, index in zip(range(j, j + self.data_nqubits), range(self.class_nqubits, self.class_nqubits + self.data_nqubits)):
                
                qc.ry(angle_y[i+j], index)
                qc.rz(angle_z[i+j], index)
                
                qc.ry(angle_y[i+j], index + self.data_nqubits + self.class_nqubits)
                qc.rz(angle_z[i+j], index + self.data_nqubits + self.class_nqubits)
            
        return qc, qr

    def U_out(self, qc, qr):
        """output用ゲートU_outの組み立て&パラメータ初期値の設定"""
        
        # time_evol_gate = create_time_evol_gate(self.nqubit - 1)
        
        for d in range(self.c_depth):
            
            # u_out.add_gate(time_evol_gate)
            
            for i in range(self.nqubit):
                qc.rx(self.weights[self.nqubit*d + i], i)
            for i in range(self.nqubit-1):
                qc.cx(i, i+1)
            if self.nqubit > 1:
                qc.cx(self.nqubit-1, 0)
            
        qc.barrier()
        
        return qc
    
    def U(self, x, y):
        """メインの量子回路"""
        
        qc, qr = self.U_in(x, y)
        qc.barrier()
        qc = self.U_out(qc, qr)
        # qc.measure_all()
        
        # circuit_drawer(qc, filename="quantum_circuit.png", output='mpl')

        return qc
    
    def update_params(self, qc, theta):
        """U_outをパラメータθで更新"""
        
        self.theta = theta
        qc = qc.bind_parameters({self.weights : theta})
        
        return qc
    
    def output(self, theta):
        """x_listに対して、モデルの出力を計算"""
        
        outputs = []
        
        simulator = AerSimulator(device='GPU')
        nshots = 10000
        q_instance = QuantumInstance(simulator, shots=nshots)
        
        # 各クラスについて期待値計算（不正解クラスも含む）
        for x, y_mixed in zip(self.x_list, self.y_mixed_list):

            qc = self.U(x, y_mixed)
            qc = self.update_params(qc, theta)
            
            psi = CircuitStateFn(qc) # 量子回路出力の状態ベクトルを取得
            
            # 期待値計算
            # define the state to sample
            measurable_expression = StateFn(self.obs, is_measurement=True).compose(psi)
            # convert to expectation value
            expectation = PauliExpectation().convert(measurable_expression)
            # get state sampler (you can also pass the backend directly)
            sampler = CircuitSampler(q_instance).convert(expectation)
            
            outputs.append(np.array([y_mixed[-1], sampler.eval().real]))
            
        outputs = np.array(outputs)
        
        return outputs

    def loss(self, is_correct, innerproduct):
        if is_correct:
            return (0.5 - innerproduct) * 2
        else:
            return innerproduct * 2

    def cost_func(self, theta):
        
        """コスト関数を計算するクラス
        :param theta: 回転ゲートの角度thetaのリスト
        """
        
        outputs = self.output(theta)       
        losses = [self.loss(is_correct, expval) for is_correct, expval in outputs]
        # バッチ処理
        loss = np.sum(losses)
        
        return loss

    # for GDSD
    # 期待値の微分
    def B_grad(self, theta):
        # dB/dθ のリストを返す
        theta_plus = [theta.copy() + np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]
        theta_minus = [theta.copy() - np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]
        
        grad = [(self.output(theta_plus[i])[:, -1] - self.output(theta_minus[i])[:, -1]) / 2. for i in range(len(theta))]

        return np.array(grad)
    
    # for cost_func_grad of GDSD 
    def cost_grad_by_B(self, theta):
        # dL/dB のリストを返す  
        bool_for_label = self.output(theta)[:, 0]  

        cost_grad_by_B = []
        
        for is_correct in bool_for_label:
            if is_correct:
                cost_grad_by_B.append(-2)
            else:
                cost_grad_by_B.append(2)

        return cost_grad_by_B

    # for GDSD
    def cost_func_grad(self, theta):
        # dL/dθ のリストを返す
        B_gr_list = self.B_grad(theta)
        cost_grad_by_B = self.cost_grad_by_B(theta)
        
        grad = [np.sum(cost_grad_by_B * B_gr) for B_gr in B_gr_list]
        
        # print("grad:\n", grad)
        # print("len(grad):\n", len(grad))
        
        return np.array(grad)
    
    def batch_generator(self, x_list, y_list, batch_size):
        
        num_samples = x_list.shape[0]
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start_idx in range(0, num_samples, batch_size):
            batch_indices = indices[start_idx: start_idx + batch_size]
            yield x_list[batch_indices], y_list[batch_indices]        
        
        
    # パラメータ更新
    def optimize_params(self, theta):
        
        lr = 1e-3
        
        grad = self.cost_func_grad(theta)
        
        update_weights = theta - lr * grad
        
        return update_weights
    
    
    def fit(self, x_list, y_list, y_mixed_list, maxiter):
        
        """
        :param x_list: fitしたいデータのxのリスト
        :param y_list: fitしたいデータのyのリスト
        :param y_mixed_list: fitしたい不正解クラスを含むデータのyのリスト
        :param maxiter: scipy.optimize.minimizeのイテレーション回数
        :return: 学習後のロス関数の値
        :return: 学習後のパラメータthetaの値
        """
        
        
        # 乱数でUを作成
        theta = 2 * np.pi * algorithm_globals.random.random(self.n_params)
        self.theta = theta.flatten()
        theta_init = self.theta
        
        theta_opt = theta_init

        # for callbacks
        self.n_iter = 0
        # self.maxiter = maxiter
        self.maxiter = self.batch_size
        
        for i in range(self.num_epochs):
            
            print(f'============================== {i+1} epoch ==============================')
            
            for x_list_batch, y_list_batch in self.batch_generator(x_list, y_mixed_list, self.batch_size):
                
                self.x_list = x_list_batch
                self.y_mixed_list = y_list_batch
                
                # print('self.x_list:\n', self.x_list)
                # print('self.x_list.shape:\n', self.x_list.shape)
                # print('self.y_mixed_list:\n', self.y_mixed_list)
                # print('self.y_mixed_list.shape:\n', self.y_mixed_list.shape)
                
                print("Initial parameter:")
                print(self.theta)
                print()
                print(f"Initial value of cost function:  {self.cost_func(self.theta):.4f}")
                print('=========================================================')
                
                print("Iteration count...")
                
                # result = minimize(self.cost_func,
                #                 self.theta,
                #                 # method='Nelder-Mead',
                #                 method='BFGS',
                #                 jac=self.cost_func_grad,
                #                 options={"maxiter":maxiter},
                #                 callback=self.callbackF)
                
                
                # 最急降下法
                self.theta = self.optimize_params(self.theta)
                self.callbackF(self.theta)
        
                theta_opt = self.theta
        
        
        print('=========================================================')
        print()
        print("Optimized parameter:")
        print(self.theta)
        print()
        print(f"Final value of cost function:  {self.cost_func(self.theta):.4f}")
        print()
            
        return theta_init, theta_opt

    def callbackF(self, theta):
        self.n_iter = self.n_iter + 1
        
        save_graph_loss(loss=self.cost_func(theta), 
                        title="Objective function value against iteration", 
                        filename=self.fig_name_loss, 
                        n_iter=self.n_iter)
        save_graph_grad(grads=self.cost_func_grad(theta), 
                        filename=self.fig_name_grad,
                        n_iter=self.n_iter,
                        show_errorbar=False)
        
        if 10 * self.n_iter % self.maxiter == 0:
            print(f"Iteration: {self.n_iter} / {self.maxiter},   Value of cost_func: {self.cost_func(theta):.4f}")
