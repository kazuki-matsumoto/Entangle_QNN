from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.utils import algorithm_globals
import numpy as np
import torch
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import ParameterVector

num_inputs = 3
qr = QuantumRegister(num_inputs)
cr = ClassicalRegister(1)

# Set seed for random generators
algorithm_globals.random_seed = 42

weights = ParameterVector('weight', 6)

feature_map = ZZFeatureMap(num_inputs-1)
qc = QuantumCircuit(qr, cr)
qc.compose(feature_map, inplace=True)
qc.barrier()
qc.ry(weights[0], 0)
qc.ry(weights[1], 1)
qc.ry(weights[2], 2)
qc.cswap(num_inputs-1, 0, 1)
qc.ry(weights[3], 0)
qc.ry(weights[4], 1)
qc.ry(weights[5], 2)
qc.measure(num_inputs-1, cr[0])

print(qc)

def identity_interpret(x):
    return x

qnn = SamplerQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=weights,
    interpret=identity_interpret,
    output_shape=2,
)

initial_weights = 0.1 * (2 * algorithm_globals.random.random(qnn.num_weights) - 1)
print("Initial weights: ", initial_weights)

def cost_func_domain(params_values):
    probabilities = qnn.forward(torch.tensor([1, 2]), params_values)
    cost = np.sum(probabilities[:, 1])
    print("cost : ", cost)
    return cost

cost_func_domain(torch.tensor(initial_weights))

opt = COBYLA(maxiter=150)
objectiv_func_val = []

opt_result = opt.minimize(cost_func_domain, initial_weights)

print(opt_result.x)