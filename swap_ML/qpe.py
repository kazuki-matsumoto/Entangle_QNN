from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import transpile, assemble
from qiskit.circuit import Parameter
import numpy as np
from qiskit.circuit.library import QFT
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# 第1レジスタのビット数
N_ENCODE = 10
# 第2レジスタのビット数
N_EIGEN_STATE = 2

N_QUBITS = N_ENCODE + N_EIGEN_STATE

# qr = QuantumRegister(N_QUBITS)
# cr = ClassicalRegister(N_ENCODE)
# qc = QuantumCircuit(qr, cr)

def inverse_qft(n):
    qft = QFT(n)
    inv_qft = qft.inverse()
    
    return inv_qft

def qpe(y):
    
    qr = QuantumRegister(N_QUBITS)
    cr = ClassicalRegister(N_ENCODE)
    qc = QuantumCircuit(qr, cr)
    
    # 位相を推定したい状態(|11>)
    y_bin = format(y, '02b')
    for i, y_i in enumerate(y_bin):
        if y_i == "1":
            qc.x(N_ENCODE+i)

    qc.barrier()
    
    for i in range(N_ENCODE):
        qc.h(i)
    
    theta = Parameter('θ')
    r = 1
    for c in range(N_ENCODE):
        for i in range(r):
            qc.cp(theta, control_qubit=c, target_qubit=N_QUBITS-1)
            qc.cp(theta, control_qubit=c, target_qubit=N_QUBITS-2)
        r *= 2
    
    qc.append(inverse_qft(N_ENCODE), qc.qubits[:N_ENCODE])
    
    # print(qc)
    
    qc.barrier()
    
    for i in range(N_ENCODE):
        qc.measure(i, i)
    
    return qc, theta

for i in range(4):
    qc, theta = qpe(i)
    simulator = QasmSimulator()
    shots = 10000

    # θ=2*pi*φを与える（正解値）
    np.random.seed(0)
    phase = np.random.rand()
    # 回路にパラメータを設定する。
    qc_para = qc.bind_parameters({theta: phase})

    compiled_circuit = transpile(qc_para, simulator)
    qobj = assemble(compiled_circuit, shots=shots)
    results = simulator.run(qobj, shots=shots).result()

    answer = results.get_counts()

    # 測定された位相の算出
    values = list(results.get_counts().values())
    keys = list(results.get_counts().keys())
    idx = np.argmax(list(results.get_counts().values()))
    ans = int(keys[idx], 2)

    phase_estimated = ans / (2 ** N_ENCODE)

    print(i)

    # 正しい位相の値
    true_phase = phase / (2 * np.pi)
    print('True phase: {:.4f}'.format(true_phase))
    print('Estimated phase: {:.4f}'.format(phase_estimated))
    print('Diff: {:.4f}'.format(np.abs(true_phase - phase_estimated)))

    # ヒストグラムを描画して保存する。
    # plot_histogram(answer, figsize=(20, 7))
    # plt.savefig("./histogram.jpg")
