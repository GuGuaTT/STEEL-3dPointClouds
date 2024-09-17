import torch.nn as nn
import torch.nn.functional as F


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.fc1 = nn.Linear(7, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.drop3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(512, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.drop4 = nn.Dropout(0.3)
        
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        
        x = self.drop1(F.leaky_relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.leaky_relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.leaky_relu(self.bn3(self.fc3(x))))
        x = self.drop4(F.leaky_relu(self.bn4(self.fc4(x))))
        x = self.fc5(x)
    
        return x
