import torch
from torch import nn
from torch.nn import functional as F
from torch import optim


class ConvNet(nn.Module):

    '''   
    (Operations)                                    (Input Size)
    -------------------------------------------------------------
    Initial input :                                  2 x 14 x 14
    Convolutional Layer with ReLU(1) :               32 x 12 x 12 
    MaxPool2D(Kernel size = 2 & Stride = 2)(1) :     32 x 6 x 6  
    BatchNorm2d(1) + Dropout(Probability = 0.4)(1) : 32 x 6 x 6
    Convolutional Layer with ReLU(2) :               64 x 4 x 4 
    MaxPool2D(Kernel size = 2 & Stride = 2)(2) :     64 x 2 x 2  
    BatchNorm2d(2) + Dropout(Probability = 0.2)(2) : 64 x 2 x 2
    Fully Connected Layer with ReLU(1) :             256 x 1 -> 128 x 1
    Fully Connected Layer with Softmax(2) :          128 x 1 -> 2 x 1
    '''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3) 
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 128)  
        self.fc2 = nn.Linear(128, 2)
       
   
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = self.dropout1(self.bn1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = self.dropout2(self.bn2(x))
        x = F.relu(self.fc1(x.view(-1, 256)))))
        return F.softmax(self.fc2(x), dim = 1)