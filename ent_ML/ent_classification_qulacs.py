import numpy as np
from qulacs import QuantumState, Observable, QuantumCircuit, ParametricQuantumCircuit
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from ent_utils import create_time_evol_gate, min_max_scaling, softmax
from sklearn.metrics import accuracy_score
from qulacsvis import circuit_drawer
from qulacs.gate import *


# --------------------------- ここを修正 ------------------------------------
class QclClassification:
    """ quantum circuit learningを用いて分類問題を解く"""
    def __init__(self, nqubit, data_nqubits, class_nqubits, num_features, c_depth, num_class, num_measure=1):
        
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

        # オブザーバブルの準備
        obs = [Observable(nqubit) for _ in range(num_measure)]
        for i in range(len(obs)):
            obs[i].add_operator(1., f'Z {i}')  # Z0, Z1, Z3をオブザーバブルとして設定
        self.obs = obs

    def create_input_gate(self, x, y):
        
        # 単一の入力x, yをエンコードするゲートを作成する関数
        # xは入力特徴量(2次元)。
        # yはオンホットベクトル
        # xの要素は[-1, 1]の範囲内
        
        u = QuantumCircuit(self.nqubit)
        
        index_y = np.argmax(y)
        
        # encode for label (y)
        for i, y_i in enumerate(format(index_y, f'0{self.class_nqubits}b')):
            if y_i == "1":
                u.add_X_gate(i)
                u.add_X_gate(i + self.data_nqubits + self.class_nqubits)
        
        angle_y = np.arcsin(x[0].astype(np.float64))
        angle_z = np.arccos((x[0]**2).astype(np.float64))
        
        # encode for input data (x)
        for j in range(self.num_features - self.data_nqubits):
            for i, index in zip(range(j, j + self.data_nqubits), range(self.class_nqubits, self.class_nqubits + self.data_nqubits)):
                
                u.add_RY_gate(index, angle_y[i+j])
                u.add_RZ_gate(index, angle_z[i+j])
                
            
                u.add_RY_gate(index + self.data_nqubits + self.class_nqubits, angle_y[i+j])
                u.add_RZ_gate(index + self.data_nqubits + self.class_nqubits, angle_z[i+j])

        return u

    def set_input_state(self, x_list, y_mixed_list):
        """入力状態のリストを作成"""
        x_list_normalized = min_max_scaling(x_list)  # xを[-1, 1]の範囲にスケール
        
        st_list = []
        
        for x, y in zip(x_list_normalized, y_mixed_list):
            st = QuantumState(self.nqubit)
            input_gate = self.create_input_gate(x, y)
            input_gate.update_quantum_state(st)
            st_list.append(st.copy())
        self.input_state_list = st_list
        
        

    def create_initial_output_gate(self):
        """output用ゲートU_outの組み立て&パラメータ初期値の設定"""
        u_out = ParametricQuantumCircuit(self.nqubit)
        time_evol_gate = create_time_evol_gate(self.nqubit - 1)
        
        theta = 2.0 * np.pi * np.random.rand(self.c_depth, self.data_nqubits + self.class_nqubits - 1, 6)
        
        self.theta = theta.flatten()
        print("theta", theta)
        
        for d in range(self.c_depth):
            u_out.add_gate(time_evol_gate)
            
            u_out.add_H_gate(self.data_nqubits + self.class_nqubits)
            
            for i in range(self.data_nqubits + self.class_nqubits - 1):
                
                u_out.add_parametric_RX_gate(i, theta[d, i, 0])
                u_out.add_parametric_RZ_gate(i, theta[d, i, 1])
                u_out.add_parametric_RX_gate(i, theta[d, i, 2])
                
                u_out.add_gate(CNOT(i, i+1))
                
                u_out.add_parametric_RX_gate(i, theta[d, i, 3])
                u_out.add_parametric_RZ_gate(i, theta[d, i, 4])
                u_out.add_parametric_RX_gate(i, theta[d, i, 5])
                
                u_out.add_gate(CNOT(i + self.data_nqubits + self.class_nqubits, 
                                    i + 1 + self.data_nqubits + self.class_nqubits))
        
        
        circuit_drawer(u_out)
        
        self.output_gate = u_out
        
        
    
    def update_output_gate(self, theta):
        """U_outをパラメータθで更新"""
        self.theta = theta
        parameter_count = len(self.theta)
        for i in range(parameter_count):
            self.output_gate.set_parameter(i, self.theta[i])

    def get_output_gate_parameter(self):
        """U_outのパラメータθを取得"""
        parameter_count = self.output_gate.get_parameter_count()
        theta = [self.output_gate.get_parameter(ind) for ind in range(parameter_count)]
        return np.array(theta)

    def pred(self, theta):
        """x_listに対して、モデルの出力を計算"""

        # 入力状態準備
        # st_list = self.input_state_list
        st_list = [st.copy() for st in self.input_state_list]  # ここで各要素ごとにcopy()しないとディープコピーにならない
        # U_outの更新
        self.update_output_gate(theta)

        res = []
        # 出力状態計算 & 観測
        for st in st_list:
            # U_outで状態を更新
            self.output_gate.update_quantum_state(st)
            # モデルの出力
            r = [o.get_expectation_value(st) for o in self.obs]  # 出力多次元ver
            r = softmax(r)
            res.append(r.tolist())
        return np.array(res)

    def cost_func(self, theta):
        """コスト関数を計算するクラス
        :param theta: 回転ゲートの角度thetaのリスト
        """

        y_pred = self.pred(theta)

        # cross-entropy loss
        loss = log_loss(self.y_list, y_pred)
        
        return loss

    # for BFGS
    def B_grad(self, theta):
        # dB/dθのリストを返す
        theta_plus = [theta.copy() + np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]
        theta_minus = [theta.copy() - np.eye(len(theta))[i] * np.pi / 2. for i in range(len(theta))]

        grad = [(self.pred(theta_plus[i]) - self.pred(theta_minus[i])) / 2. for i in range(len(theta))]

        return np.array(grad)

    # for BFGS
    def cost_func_grad(self, theta):
        y_minus_t = self.pred(theta) - self.y_list
        B_gr_list = self.B_grad(theta)
        grad = [np.sum(y_minus_t * B_gr) for B_gr in B_gr_list]
        return np.array(grad)
    
    def accuracy_score(self, theta_opt, x_test, y_test):
        # 初期状態生成
        self.set_input_state(x_test)
        
        y_pred = self.pred(theta_opt)
        
        # print("y_pred", y_pred)
        # print("y_pred.size", y_pred.size)
        # print("y_test", y_test)
        # print("y_test.size", y_test.size)
        
        correct = 0
        total = 0
        
        for i in range(len(y_test)):
            act_label = np.argmax(y_test[i])
            pred_label = np.argmax(y_pred[i])
            
            if act_label == pred_label:
                correct += 1
            
            total += 1
        
        accuracy = correct/total
        
        return accuracy
        
    
    def fit(self, x_list, y_list, y_mixed_list ,maxiter=1000):
        """
        :param x_list: fitしたいデータのxのリスト
        :param y_list: fitしたいデータのyのリスト
        :param maxiter: scipy.optimize.minimizeのイテレーション回数
        :return: 学習後のロス関数の値
        :return: 学習後のパラメータthetaの値
        """

        # 初期状態生成
        self.set_input_state(x_list, y_mixed_list)

        # 乱数でU_outを作成
        self.create_initial_output_gate()
        theta_init = self.theta

        # 正解ラベル
        self.y_list = y_list

        # for callbacks
        self.n_iter = 0
        self.maxiter = maxiter
        
        print("Initial parameter:")
        print(self.theta)
        print()
        print(f"Initial value of cost function:  {self.cost_func(self.theta):.4f}")
        print()
        print('============================================================')
        print("Iteration count...")
        result = minimize(self.cost_func,
                          self.theta,
                          # method='Nelder-Mead',
                          method='BFGS',
                          jac=self.cost_func_grad,
                          options={"maxiter":maxiter},
                          callback=self.callbackF)
        theta_opt = self.theta
        print('============================================================')
        print()
        print("Optimized parameter:")
        print(self.theta)
        print()
        print(f"Final value of cost function:  {self.cost_func(self.theta):.4f}")
        print()
        return result, theta_init, theta_opt

    def callbackF(self, theta):
            self.n_iter = self.n_iter + 1
            if 10 * self.n_iter % self.maxiter == 0:
                print(f"Iteration: {self.n_iter} / {self.maxiter},   Value of cost_func: {self.cost_func(theta):.4f}")
