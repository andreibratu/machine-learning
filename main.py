import torch
from models.lenet import LeNet
import torch.nn as nn
import torch.optim as optim

net = LeNet()
print(net)

input = torch.randn(1, 1, 32, 32)
out = net(input)
net.zero_grad()

target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()