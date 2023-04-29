from qiskit_aer import AerError
from qiskit import Aer

# Initialize a GPU backend
# Note that the cloud instance for tutorials does not have a GPU
# so this will raise an exception.
try:
    simulator_gpu = Aer.get_backend('aer_simulator')
    simulator_gpu.set_options(device='GPU')
    print('successed')
except AerError as e:
    print(e)