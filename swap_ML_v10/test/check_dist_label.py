from qiskit import *
from qiskit_aer import AerSimulator

n_qubits = 5
n_shots = 10000

qr = QuantumRegister(n_qubits)
cr = ClassicalRegister(1)

qc = QuantumCircuit(qr, cr)

def swap_test(qc, q1, q2):
    
    y1_bin = format(q1, '02b')
    y2_bin = format(q2, '02b')
    
    for i, y in enumerate(y1_bin):
        if y == "1":
            qc.x(i)
    for i, y in enumerate(y2_bin):
        if y == "1":
            qc.x(i+2)
    
    qc.barrier()
    qc.h(n_qubits-1)
    qc.cswap(n_qubits-1, 0, 2)
    qc.cswap(n_qubits-1, 1, 3)
    qc.h(n_qubits-1)
    qc.barrier()
    
    
    qc.measure(n_qubits-1, 0)
    
    return qc

swapTest = swap_test(qc, 1, 3)
print(swapTest)

simulator = AerSimulator()
compiled_circuit = transpile(swapTest, simulator)
qobj = assemble(compiled_circuit, shots=n_shots)
results = simulator.run(qobj, shots=n_shots).result()
answer = results.get_counts()
print(answer)



