import torch
from torch import nn

from CNN import ConvClassificationNet
from FC import DenseClassificationNet


class AuxiliaryNet(nn.Module):

    '''   
    UTILIZE SOFTMAX DISTRIBUTIONS AS INPUT TO FULLY CONNECTED NETWORK
    (Operations)                                    (Input Size)
    -------------------------------------------------------------
    Initial input :                                  2 x 14 x 14
    Seperate it from the first dimension:            1 x 14 x 14 (Both digits) 
    Classify with Convolutional Networks:            1 x 14 x 14 -> 10 x 1 (Both digits)
    Stack digits:                                    10 x 1 and 10 x 1 -> 20 x 1
    Classify with Dense Network:                     20 x 1 -> 2 x 1

    '''

    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.convclassnet = ConvClassificationNet()
        self.denseclassnet = DenseClassificationNet()

    def forward(self, x):
        # We use dimension 1 since dimension 0 is for the batch size, i.e. our input is something like 100x2x14x14
        digit1, digit2 = x.split(split_size=1, dim=1)
        digit1 = self.convclassnet(digit1)
        digit2 = self.convclassnet(digit2)
        digits_stacked = torch.hstack((digit1, digit2))
        return self.denseclassnet(digits_stacked), digit1, digit2
