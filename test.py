import dlc_practical_prologue as prologue
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
    1000)

print('train_input', train_input.size(), 'train_target',
      train_target.size(), 'train classses', train_classes.size())
print('test_input', test_input.size(), 'test_target',
      test_target.size(), 'test classses', test_classes.size())


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, 3, 1)  # 2x14x14 -> 32x12x12
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)  # 32x12x12 -> 64x10x10
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(8*8*64, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # -1
        x = self.dropout1(x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # -1
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        # binary class
        return F.softmax(self.fc2(x), dim=1)


if __name__ == "__main__":
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_input) for i in range(n_epochs + 1)]
    network = ConvNet()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_input):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_input),
                    100. * batch_idx / len(train_input), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(
                    (batch_idx*64) + ((epoch-1)*len(train_input)))
                torch.save(network.state_dict(), '/results/model.pth')
                torch.save(optimizer.state_dict(), '/results/optimizer.pth')

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_input:
                output = network(data)
                test_loss += F.nll_loss(output, target,
                                        size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_input.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_input.dataset),
            100. * correct / len(test_input.dataset)))

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()
