import dlc_practical_prologue as prologue
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(1000)

print('train_input', train_input.size(), 'train_target', train_target.size(), 'train classses',train_classes.size())
print('test_input', test_input.size(), 'test_target', test_target.size(), 'test classses',test_classes.size())


class ConvNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)