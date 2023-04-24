from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np

nqubits = 12
qr = QuantumRegister(nqubits)
qc = QuantumCircuit(qr)

# ラベルは基底エンコーディング、入力データは振幅エンコーディング
def U_in(x, y, nqubits):
    y_bin = format(y, '0'+str(nqubits)+'b')
    
    print(y_bin)
    
    # ラベルのエンコード
    for i, y in enumerate(y_bin):
        print(y)
        if y == "1":
            qc.x(i)
            qc.x(i+7)
    
    # 入力データのエンコード
    angle_y = np.arcsin(x)
    angle_z = np.arccos(x**2)
    
    for i in range(5, 5+2):
        qc.ry(angle_y, i)
        qc.rz(angle_z, i)
        
        qc.ry(angle_y, i+4)
        qc.rz(angle_z, i+4)
    
        
    print(qc)
    


U_in(1, 3, 2)
# num = 2

# num_bin = format(num, '02b')

# print(num_bin)