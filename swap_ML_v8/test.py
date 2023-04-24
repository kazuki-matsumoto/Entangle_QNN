import qiskit as qk
from qiskit import Aer
import torch

backend = Aer.get_backend('qasm_simulator')
q = qk.QuantumRegister(1)
c = qk.ClassicalRegister(1)
theta = torch.tensor([0.5], requires_grad=True)
theta_optim = theta
optimizer = torch.optim.Adam([theta_optim], lr=0.1)
target_prob = torch.tensor([0.5])
loss_fn = torch.nn.MSELoss()
num_iterations = 1000

for i in range(num_iterations):
    circuit = qk.QuantumCircuit(q, c)
    circuit.ry(theta_optim.item(), q[0])
    circuit.measure(q, c)
    job = qk.execute(circuit, backend, shots=1000)
    
    result = job.result()
    counts = result.get_counts(circuit)
    counts_0 = counts['0'] if '0' in counts else 0
    counts_1 = counts['1'] if '1' in counts else 0
    measured_prob = counts_0 / (counts_0 + counts_1)
    
    loss = loss_fn(torch.tensor([measured_prob], requires_grad=True), target_prob)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Iteration {i+1}, theta={theta_optim.item()}, measured_prob={measured_prob}, loss={loss.item()}")
