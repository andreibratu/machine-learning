import torch.nn as nn
import torch.nn.functional as F

DEBUG = True


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 3 input image channels, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        # An affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 248)  # 6*6 from image dimension
        self.fc2 = nn.Linear(248, 124)
        self.fc3 = nn.Linear(124, 84)
        self.fc4 = nn.Linear(84, 10)

    def forward(self, x):
        global DEBUG
        # Max pooling over a (2, 2) window
        if DEBUG: print(x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        if DEBUG: print(x.size())
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        if DEBUG: print(x.size())
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        DEBUG = False
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
