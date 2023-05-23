from qiskit import *
import numpy as np
from qiskit_aer import AerSimulator
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
import mthree
from qiskit.providers.fake_provider import FakeAthens

# 量子回路の定義
def make_circuit(n_qubits, reps, thetas):
    ansatz = QuantumCircuit(n_qubits)
    for r in range(reps):
        for i in range(n_qubits):
            ansatz.rx(thetas[n_qubits*r+i], i)
        for i in range(n_qubits-1):
            ansatz.cx(i, i+1)
        if n_qubits > 1:
            ansatz.cx(n_qubits-1, 0)
    
    ansatz.measure_all()
    
    return ansatz

# 位相を、[0, 2π]に入るように正規化
def normalize_phase(phase):
    return ((phase/(2*np.pi))%1) * 2*np.pi

# # 量子回路を実行し、期待値を計算
# def get_expectation_value(qc):
#     simulator = AerSimulator(device='GPU')
#     trans_qc = transpile(qc, simulator)
    
#     trans_qc.save_statevector()
#     job = simulator.run(trans_qc, shots=1000)
#     statevector = job.result().get_statevector(trans_qc)
    
#     return float(statevector[0].real)

# 量子回路を実行し、期待値を計算
def get_expectation_value(qc, n_qubits):
    simulator = AerSimulator(device='GPU')
    trans_qc = transpile(qc, simulator)
    
    raw = simulator.run(trans_qc, shots=4000).result().get_counts()
    
    mit = mthree.M3Mitigation(simulator)
    mit.cals_from_system()
    
    quasi = mit.apply_correction(raw, np.arange(n_qubits), return_mitigation_overhead=True)
    
    return quasi.expval()

# 勾配降下法によるパラメータを更新
def update_thetas(n_qubits, reps, thetas, lr=0.01):
    new_thetas = thetas[:]
    grads = [0]*len(thetas)
    # パラメータシフト法
    for i, theta in enumerate(thetas):
        theta2 = thetas[:]
        theta2[i] = normalize_phase(theta + np.pi/2)
        ev2 = get_expectation_value(make_circuit(n_qubits, reps, theta2), n_qubits)
        
        theta3 = thetas[:]
        theta3[i] = normalize_phase(theta - np.pi/2)
        ev3 = get_expectation_value(make_circuit(n_qubits, reps, theta3), n_qubits)
        
        grad = ev2 - ev3
        new_thetas[i] = theta - lr * grad
        grads[i] = grad
    
    return new_thetas, grads
        

def do_experiment(n_qubits, n_epoch, reps=2):
    initial_weights = 2 * np.pi * algorithm_globals.random.random(reps*n_qubits)
    
    expectation_values = []
    grads_list = []
    thetas = initial_weights[:]
    
    for epoch in range(n_epoch):
        qc = make_circuit(n_qubits, reps, thetas)
        ev = get_expectation_value(qc, n_qubits)
        expectation_values.append(ev)
        thetas, grads = update_thetas(n_qubits, reps, thetas)
        grads_list.append(grads)

    return expectation_values, grads_list

def plot_expectation_values(n_qubits, n_epoch, filename):
    expactation_values, _ = do_experiment(n_qubits, n_epoch=n_epoch)
    
    x = np.arange(n_epoch) + 1
    y = expactation_values[:]
    
    plt.plot(x, y, label="{0}qubits".format(n_qubits))
    
    plt.xlabel('epochs')
    plt.ylabel('expectation value')
    
    # 凡例の表示
    plt.legend()

    # プロット表示(設定の反映)
    plt.show()
    
    plt.title('Experiment {0} qubits'.format(n_qubits))
    
    plt.savefig("test_fig_EV_vs_epoch/{0}.png".format(filename))


def plot_graph(n_qubits2grads, filename, yscale_value='linear', show_errorbar=True):
    n_qubits_range = range(1, len(n_qubits2grads)+1)
    mean_abs_grads = []
    std_abs_grads = []
    for n_qubits, grads_list in sorted(n_qubits2grads.items(), key=lambda k_v: k_v[0]):
        abs_grads = []
        for grads in grads_list:
            for grad in grads:
                abs_grads.append(abs(grad))
        mean_abs_grads.append(np.mean(abs_grads))
        std_abs_grads.append(np.std(abs_grads))

    fig, ax = plt.subplots()
    yerr = std_abs_grads if show_errorbar else None
    ax.errorbar(x=n_qubits_range, y=mean_abs_grads, yerr=yerr, fmt='-o', color='b')
    ax.set_xlabel('num of qubits')
    ax.set_ylabel('mean abs grads')
    ax.set_title('mean abs grads per each epoch')
    ax.set_yscale(yscale_value)
    plt.grid()
    plt.savefig('n_qubits2grads/{0}.png'.format(filename))


for n_qubits in [1, 2, 6, 7, 8]:
    n_epoch=100
    filename = "{0}qubits_ev_vs_epoch".format(n_qubits)
    plot_expectation_values(n_qubits, n_epoch, filename)

n_qubits2grads = {}

for n_qubit in range(1, 25+1):
    expactation_values, grads_list = do_experiment(n_qubit, n_epoch=10)
    n_qubits2grads[n_qubit] = grads_list[:10]
    
    print("n_qubits2grads", n_qubits2grads)

plot_graph(n_qubits2grads, yscale_value='linear', filename='n_qubits2grads_linear')
plot_graph(n_qubits2grads, yscale_value='log', filename='n_qubits2grads_log')