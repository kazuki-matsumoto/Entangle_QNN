import torch
import numpy as np

x_0 = torch.tensor(np.array([0., 0.]), requires_grad=True, device=torch.device('cuda'))
x = x_0
optimizer = torch.optim.SGD([x], lr=0.1)
steps = 30

for i in range(steps):
    optimizer.zero_grad()
    f = x**2
    f.backward()
    optimizer.step()
    
    print(f'At step {i+1:2} the function value is {f.item(): 1.4f} and x={x: 0.4f} ')
    
