import torch
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZFeatureMap, RealAmplitudes

from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN

num_qubits = 2
fmap = ZFeatureMap(num_qubits, reps=1)
ansatz = RealAmplitudes(num_qubits, reps=1)
qc = QuantumCircuit(num_qubits)
qc.compose(fmap, inplace=True)
qc.compose(ansatz, inplace=True)

qnn = SamplerQNN(
    circuit=qc,
    input_params=fmap.parameters,
    weight_params=ansatz.parameters,
    sparse=True,
)

connector = TorchConnector(qnn)

output = connector(torch.tensor([[1., 2.]]))
print(output)

loss = torch.sparse.sum(output)
loss.backward()

grad = connector.weight.grad
print(grad)