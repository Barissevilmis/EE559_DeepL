import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from CNN import ConvNet
from FC import DenseNet


class ConvDenseNet(nn.Module):

    '''   
    Merging Convolutional and Dense Networks
    (Operations)                                    (Input Size)
    -------------------------------------------------------------
    Initial input :                                  2 x 14 x 14
    
    '''

    def __init__(self):
        super(ConvDenseNet, self).__init__()
        self.convnet = ConvNet()
        self.fcnet = DenseNet()
       
   
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = self.dropout1(self.bn1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)
        x = self.dropout2(self.bn2(x))
        x = F.relu(self.fc1(x.view(-1, 256)))))
        return F.softmax(self.fc2(x), dim = 1)