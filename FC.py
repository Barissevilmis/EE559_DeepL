import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class DenseNet(nn.Module):

    '''   
    (Operations)                                    (Input Size)
    -------------------------------------------------------------
    Initial input :                                  2 x 14 x 14 -> 392 x 1
    Fully Connected Layer with ReLU(1) :             392 x 1 -> 256 x 1 
    BatchNorm1d(1) + Dropout(Probability = 0.4)(1) : 256 x 1
    Fully Connected Layer with ReLU(2) :             256 x 1 -> 128 x 1 
    BatchNorm1d(2) + Dropout(Probability = 0.2)(2) : 128 x 1
    Fully Connected Layer with ReLU(3) :             128 x 1 -> 64 x 1
    BatchNorm1d(3) :                                 64 x 1   
    Fully Connected Layer with Softmax(4) :          64 x 1 -> 2 x 1
    '''

    def __init__(self):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Linear(392, 256) 
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)  
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 2)
       
   
    def forward(self, x):
        x = F.relu(self.fc1(x.view(-1, 392)))
        x = self.dropout1(self.bn1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(self.bn2(x))
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        return F.softmax(self.fc4(x), dim = 1)