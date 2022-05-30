import torch.nn as nn
import torch

class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3*32*32, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 80)
        self.fc4 = nn.Linear(80, 10)
        #self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        #x = self.out(x)
        return x