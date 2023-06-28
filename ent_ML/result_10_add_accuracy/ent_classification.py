import numpy as np
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from ent_utils import (create_time_evol_gate, 
                       min_max_scaling, 
                       softmax, 
                       save_graph_loss, 
                       save_graph_grad,
                       save_graph_accuracy)
from sklearn.metrics import accuracy_score

from qiskit.quantum_info.operators import Operator, Pauli
from qiskit import *
from qiskit.circuit import ParameterVector
from qiskit.utils import algorithm_globals
from qiskit.primitives import BackendEstimator
from qiskit_aer import AerSimulator
from qiskit.visualization import circuit_drawer


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
        fig_name_accuracy,
        num_epochs,
        batch_size,
        num_measure=1):
        
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
        
        self.n_params = 12 * (self.data_nqubits + self.class_nqubits - 1) * self.c_depth # パラメータの数
        self.weights = ParameterVector('weight', self.n_params) # パラメータのベクトル
        
        self.fig_name_loss = fig_name_loss
        self.fig_name_grad = fig_name_grad
        self.fig_name_accuracy = fig_name_accuracy
        
        backend = AerSimulator()
        self.estimator = BackendEstimator(backend)
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        # オブザーバブルの準備
        obs = []
        for i in range(num_measure):
            obs.append(Pauli('Z', num_measure - i - 1))    
        
        self.obs = obs

    def U_in(self, x, y):
        
        # 単一の入力x, yをエンコードするゲートを作成する関数
        # xは入力特徴量(2次元)。
        # yはオンホットベクトル
        # xの要素は[-1, 1]の範囲内
        
        qr = QuantumRegister(self.nqubit, 'qr')
        cr = ClassicalRegister(1, 'cr')
        qc = QuantumCircuit(qr, cr)
        
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


    def add_paramed_ent_gate(self, qc, id_qubit, skip_id1, skip_id2, depth):
        
        # depth_index =  int(self.n_params-self.n_params/depth)
        depth_index = int((self.n_params / self.c_depth) * depth)
        base_index = id_qubit * 12 + depth_index

        # Apply u gates
        qc.u(self.weights[base_index], self.weights[base_index + 1], self.weights[base_index + 2], id_qubit + skip_id1)
        qc.u(self.weights[base_index + 3], self.weights[base_index + 4], self.weights[base_index + 5], id_qubit + skip_id2)

        # Apply cx gate
        qc.cx(id_qubit + skip_id1, id_qubit + skip_id2)

        # Apply u gates
        qc.u(self.weights[base_index + 6], self.weights[base_index + 7], self.weights[base_index + 8], id_qubit + skip_id1)
        qc.u(self.weights[base_index + 9], self.weights[base_index + 10], self.weights[base_index + 11], id_qubit + skip_id2)
 
 
    # パラメータ化量子回路
    def parametarized_qcl(self, depth):
        
        n_qubits = self.class_nqubits + self.data_nqubits
        
        para_ent_qr = QuantumRegister(n_qubits)
        para_ent_qc = QuantumCircuit(para_ent_qr, name="Parametarized Gate")

        for id_qubit in range(n_qubits-1):
            
            self.add_paramed_ent_gate(para_ent_qc, id_qubit, 0, 1, depth)
            
            para_ent_qc.barrier()
        
        inst_paramed_gate = para_ent_qc.to_instruction()
        
        return inst_paramed_gate
    
    # 最大エンタングルメント状態
    def max_entanglement_gate(self):
        
        n_qubits = self.class_nqubits + self.data_nqubits
        
        max_ent_qr = QuantumRegister(n_qubits)
        max_ent_qc = QuantumCircuit(max_ent_qr, name="Max Entanglement Gate")
        
        max_ent_qc.h(0)
        
        for i in range(n_qubits-1):
            max_ent_qc.cx(i, i+1)
        inst_max_ent = max_ent_qc.to_instruction()
        
        return inst_max_ent

    def U_out(self, qc, qr):
        """output用ゲートU_outの組み立て&パラメータ初期値の設定"""
        
        # time_evol_gate = create_time_evol_gate(self.nqubit - 1)
        
        for d in range(self.c_depth):
            
            # u_out.add_gate(time_evol_gate)
            
            qc.append(self.parametarized_qcl(d), [qr[i] for i in range(0, self.class_nqubits + self.data_nqubits)])
        
        qc.append(self.max_entanglement_gate(), [qr[i] for i in range(self.class_nqubits + self.data_nqubits, self.nqubit - 1)])
        
        # スワップテスト    
        qc.h(self.nqubit-1)
        for id in range(self.class_nqubits + self.data_nqubits):
            qc.cswap(self.nqubit-1, self.nqubit-2-id, self.nqubit-2-(self.class_nqubits + self.data_nqubits)-id)
        qc.h(self.nqubit-1)
        
        qc.barrier()
        
        return qc
    
    def U(self, x, y):
        """メインの量子回路"""
        
        qc, qr = self.U_in(x, y)
        qc.barrier()
        qc = self.U_out(qc, qr)
        qc.measure(self.nqubit-1, 0)
        
        # circuit_drawer(qc, filename="quantum_circuit.png", output='mpl')

        return qc
    
    def update_params(self, qc, theta):
        """U_outをパラメータθで更新"""
        
        self.theta = theta
        qc = qc.bind_parameters({self.weights : theta})
        
        return qc
    
    def output(self, theta, x_list=None, y_list=None):
        """x_list, y_listに対して、モデルの出力を計算"""
        
        innerproducts = []
        
        simulator = AerSimulator(device='GPU')
        nshots = 10000
        
        if x_list is not None and y_list is not None:
            _x_list = x_list
            _y_list = y_list
            
        else:
            _x_list = self.x_list
            _y_list = self.y_mixed_list
        
        # 各クラスについて内積計算（不正解クラスも含む）
        for x, y in zip(_x_list, _y_list):

            qc = self.U(x, y)
            qc = self.update_params(qc, theta)
            
            qc = transpile(qc, simulator)
            result = simulator.run(qc, shots=nshots).result()
            counts = result.get_counts(qc)
            
            if '1' in counts:
                b = counts['1']
            else:
                b = 0
            
            innerproduct = 1 - (2 / nshots) * b
            
            innerproducts.append(np.array([y[-1], innerproduct]))
            
        innerproducts = np.array(innerproducts)
        
        return innerproducts

    def loss(self, is_correct, innerproduct):
        if is_correct:
            return (0.5 - innerproduct) * 2
        else:
            return innerproduct * 2

    def cost_func(self, theta):
        
        """コスト関数を計算するクラス
        :param theta: 回転ゲートの角度thetaのリスト
        """
        
        y_innerproducts = self.output(theta)
        losses = [self.loss(is_correct, innerproduct) for is_correct, innerproduct in y_innerproducts]
        loss = np.sum(losses)
        
        return loss

    # for BFGS
    # 期待値の微分
    def B_grad(self, theta):
        # dB/dθ のリストを返す
        theta_plus = [theta.copy() + np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]
        theta_minus = [theta.copy() - np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]
        
        grad = [(self.output(theta_plus[i])[:, -1] - self.output(theta_minus[i])[:, -1]) / 2. for i in range(len(theta))]

        return np.array(grad)
    
    # for cost_func_grad of BFGS 
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

    # for BFGS
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
    
    
    # 予測
    def predict(self, theta, X_test_list=None, y_test_list=None):
        
        innerproducts = []
        
        print('----------------------------- predict -----------------------------')
        
        test_classes = np.array(range(self.num_class))
        bool_list = np.array([True]*self.num_class)
        test_case_with_bool = np.concatenate((np.eye(self.num_class)[test_classes], bool_list[:, np.newaxis]), axis=1)
        
        for x in X_test_list:
            block_innerproduct = []
            for y in test_case_with_bool:
                innerproduct = self.output(theta, x_list=np.array([x]), y_list=np.array([y]))
                block_innerproduct.append(innerproduct[0][-1])
                
            block_innerproduct = np.array(block_innerproduct)
            innerproducts.append(block_innerproduct)
        
        innerproducts = np.array(innerproducts)
        argmaxed_innerproducts = np.argmax(innerproducts, axis=1)
        argmaxed_y_test_list = np.argmax(y_test_list, axis=1)
        
        accuracy = accuracy_score(argmaxed_y_test_list, argmaxed_innerproducts)
        
        print("accuracy : ", accuracy)
        
        return accuracy
 
    
    
    
    
    def fit(self, x_list, y_list, y_mixed_list, x_test, y_test, maxiter):
        
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
        self.maxiter = self.num_epochs
        
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
                
                ## 最急降下法
                self.theta = self.optimize_params(self.theta)
                
                self.callbackF(self.theta, x_test, y_test)
        
                theta_opt = self.theta
        
        
        print('=========================================================')
        print()
        print("Optimized parameter:")
        print(self.theta)
        print()
        print(f"Final value of cost function:  {self.cost_func(self.theta):.4f}")
        print()
            
        return theta_init, theta_opt

    def callbackF(self, theta, x_test, y_test):
        self.n_iter = self.n_iter + 1
        
        save_graph_loss(loss=self.cost_func(theta), 
                        title="Objective function value against iteration", 
                        filename=self.fig_name_loss, 
                        n_iter=self.n_iter)
        save_graph_grad(grads=self.cost_func_grad(theta), 
                        filename=self.fig_name_grad,
                        n_iter=self.n_iter,
                        show_errorbar=False)
        save_graph_accuracy(accuracy=self.predict(theta, x_test, y_test),
                            filename=self.fig_name_accuracy,
                            n_iter=self.n_iter)
        
        if 10 * self.n_iter % self.maxiter == 0:
            print(f"Iteration: {self.n_iter} / {self.maxiter},   Value of cost_func: {self.cost_func(theta):.4f}")
