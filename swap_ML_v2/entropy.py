import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile
from qiskit.quantum_info import partial_trace, entropy
from qiskit_aer import AerSimulator

c = 1 / np.sqrt(2)

def main():
    qr = QuantumRegister(2)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    # qc.ry(2 * np.arccos(c), 0)
    qc.cx(0, 1)

    print(qc)

    backend = AerSimulator(device='GPU')
    qc = transpile(qc, backend)
    qc.save_statevector()

    job = backend.run(qc)
    statevector = job.result().get_statevector(qc)
    
    print(statevector)

    reduced_state = partial_trace(statevector, [1])
    
    print(reduced_state)
    
    print(np.array(reduced_state))
    print(entropy(reduced_state))


if __name__ == '__main__':
    main()