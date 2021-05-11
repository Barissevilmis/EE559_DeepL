import torch
from torch import nn
from torch.nn import functional as F
from torch import optim

from CNN import ConvClassificationNet
from FC import DenseClassificationNet


class HybridNet(nn.Module):

    '''   
    UTILIZE SOFTMAX DISTRIBUTIONS AS INPUT TO FULLY CONNECTED NETWORK
    (Operations)                                    (Input Size)
    -------------------------------------------------------------
    Initial input :                                  2 x 14 x 14
    Seperate it from the first dimension:            1 x 14 x 14 (Both digits) 
    Classify with Convolutional Networks:            1 x 14 x 14 -> 10 x 1 (Both digits)
    Transpose Convolution:                           10 x 1 and 10 x 1 -> 19 x 1
    Classify with Dense Network:                     19 x 1 -> 2 x 1
    
    '''

    def __init__(self):
        super(HybridNet, self).__init__()
        self.convclassnet = ConvClassificationNet()
        self.denseclassnet = DenseClassificationNet()
       
   
    def forward(self, x):
        digit1, digit2 = x.split(split_size = 1, dim = 1)
        digit1  = self.convclassnet(digit1)
        digit2  = self.convclassnet(digit2)
        return self.denseclassnet(F.conv_transpose1d(digit1, digit2))