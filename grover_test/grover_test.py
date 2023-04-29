from qiskit import *
from qiskit_aer import AerSimulator


def grover_diffusion():
        n_qubits = 2
        diffusion_qr = QuantumRegister(n_qubits)
        diffusion_qc = QuantumCircuit(diffusion_qr, name='diffusion')
        
        diffusion_qc.h(range(0, 2))
        diffusion_qc.x(range(0, 2))
        diffusion_qc.h(1)
        diffusion_qc.cx(0, 1)
        diffusion_qc.h(1)
        diffusion_qc.x(range(0, 2))
        diffusion_qc.h(range(0, 2))
        
        # convert to a gate
        inst_diff = diffusion_qc.to_instruction()
        
        return inst_diff
    
def grover(label):
    n_qubits = 5
    grover_cr = ClassicalRegister(2)
    grover_qr = QuantumRegister(n_qubits)
    grover_qc = QuantumCircuit(grover_qr, grover_cr, name='Glover Gate')
    
    label_bin = format(label, '02b')
    print(label_bin)
    for i, y in enumerate(label_bin):
        if y == "1":
            grover_qc.x(i)
    
    grover_qc.h(range(2, 4))
    grover_qc.x(4)
    grover_qc.h(4)
    grover_qc.cx(1, 3)
    grover_qc.cx(0, 2)
    
    grover_qc.barrier()
    grover_qc.x(range(2, 4))
    grover_qc.ccx(2, 3, 4)
    grover_qc.x(range(2, 4))
    grover_qc.barrier()
    
    grover_qc.cx(0, 2)
    grover_qc.cx(1, 3)
    
    grover_qc.barrier()
    
    grover_qc.append(grover_diffusion(), [grover_qr[2], grover_qr[3]])
    
    grover_qc

    return grover_qc

for label in range(4):

    print(label)
    
    grover_qc = grover(label=label)

    # simlator
    simulator = AerSimulator(device='GPU')
    # num shots
    nshots = 10000

    grover_qc.measure([2, 3], [0, 1])
    # print(grover_qc)

    compiled_circuit = transpile(grover_qc, simulator)
    job = simulator.run(compiled_circuit, shots=nshots)
    result = job.result()

    counts = result.get_counts(compiled_circuit)
    print("\nTotal count are:", counts)