import numpy as np
from functools import reduce
from qulacs.gate import X, Z, DenseMatrix
import matplotlib.pyplot as plt


# 基本ゲート
I_mat = np.eye(2, dtype=complex)
X_mat = X(0).get_matrix()
Z_mat = Z(0).get_matrix()

loss_func_vals = []
loss_x_label = []

mean_abs_grads = []
std_abs_grads = []
grads_x_label_for_grad= []

accuracy_vals = []
grads_x_label_for_accuracy = []


# fullsizeのgateをつくる関数.
def make_fullgate(list_SiteAndOperator, nqubit):
    """
    list_SiteAndOperator = [ [i_0, O_0], [i_1, O_1], ...] を受け取り,
    関係ないqubitにIdentityを挿入して
    I(0) * ... * O_0(i_0) * ... * O_1(i_1) ...
    という(2**nqubit, 2**nqubit)行列をつくる.
    """
    list_Site = [SiteAndOperator[0] for SiteAndOperator in list_SiteAndOperator]
    list_SingleGates = []  # 1-qubit gateを並べてnp.kronでreduceする
    cnt = 0
    for i in range(nqubit):
        if i in list_Site:
            list_SingleGates.append( list_SiteAndOperator[cnt][1] )
            cnt += 1
        else:  # 何もないsiteはidentity
            list_SingleGates.append(I_mat)

    return reduce(np.kron, list_SingleGates)


def create_time_evol_gate(nqubit, time_step=0.77):
    
    """ ランダム磁場・ランダム結合イジングハミルトニアンをつくって時間発展演算子をつくる
    :param time_step: ランダムハミルトニアンによる時間発展の経過時間
    :return  qulacsのゲートオブジェクト
    """
    
    ham = np.zeros((2**nqubit,2**nqubit), dtype = complex)
    for i in range(nqubit):  # i runs 0 to nqubit-1
        Jx = -1. + 2.*np.random.rand()  # -1~1の乱数
        ham += Jx * make_fullgate( [ [i, X_mat] ], nqubit)
        for j in range(i+1, nqubit):
            J_ij = -1. + 2.*np.random.rand()
            ham += J_ij * make_fullgate ([ [i, Z_mat], [j, Z_mat]], nqubit)

    # 対角化して時間発展演算子をつくる. H*P = P*D <-> H = P*D*P^dagger
    diag, eigen_vecs = np.linalg.eigh(ham)
    time_evol_op = np.dot(np.dot(eigen_vecs, np.diag(np.exp(-1j*time_step*diag))), eigen_vecs.T.conj())  # e^-iHT

    # qulacsのゲートに変換
    time_evol_gate = DenseMatrix([i for i in range(nqubit)], time_evol_op)

    return time_evol_gate


def min_max_scaling(x, axis=None):
    """[-1, 1]の範囲に規格化"""
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    result = 2.*result-1.
    return result


def softmax(x):
    """softmax function
    :param x: ndarray
    """
    exp_x = np.exp(x)
    y = exp_x / np.sum(np.exp(x))
    return y


def save_graph_loss(loss, title, filename, n_iter):
    fig1, ax1 = plt.subplots()
    
    ax1.set_title(title)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("LOSS value")
    
    loss_x_label.append(n_iter)
    loss_func_vals.append(loss)
    ax1.plot(loss_x_label, loss_func_vals)
    
    plt.savefig(filename)

    plt.close()
    

def save_graph_grad(grads, filename, n_iter, yscale_value='linear', show_errorbar=True):
    
    grads_x_label_for_grad.append(n_iter)
    mean_abs_grads.append(np.mean(abs(grads)))
    std_abs_grads.append(np.std(abs(grads)))
    
    fig, ax = plt.subplots()
    yerr = std_abs_grads if show_errorbar else None
    ax.errorbar(x=grads_x_label_for_grad, y=mean_abs_grads, yerr=yerr, fmt='-o', color='b')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('mean abs grads')
    ax.set_title('mean abs grads per each iter')
    ax.set_yscale(yscale_value)
    plt.grid()
    plt.savefig(filename)
    
    plt.close()
    

def save_graph_accuracy(accuracy, filename, n_iter):
    
    accuracy_vals.append(accuracy)
    grads_x_label_for_accuracy.append(n_iter)
    
    fig2, ax2 = plt.subplots()
    ax2.set_title('Accuracy values against iteration')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Accuracy value")
    ax2.plot(x=grads_x_label_for_accuracy, y=accuracy_vals)
    plt.grid()
    plt.savefig(filename)
    
    plt.close()