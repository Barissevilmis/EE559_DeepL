import dlc_practical_prologue as prologue
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size = 3)  # 2x14x14 -> 32x12x12
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3)  # 32x12x12 -> 64x10x10
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

    #Import data
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(
    1000)

    print('train_input', train_input.size(), 'train_target',
        train_target.size(), 'train classses', train_classes.size())
    print('test_input', test_input.size(), 'test_target',
        test_target.size(), 'test classses', test_classes.size())

    #Hyperparameters
    n_epochs = 50
    batch_size_train = 128
    learning_rate = 1e-2

    train_losses = []
    test_losses = []
    network = ConvNet()
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(zip(train_input, train_target)):
            #Model training part
            optimizer.zero_grad()
            output = network(data)
            loss = nn.BCELoss(output, target)
            loss.backward()
            optimizer.step()
            #Keep all losses
            train_losses.append(loss.item())
            print("Epoch",str(epoch)," Batch",str(batch_idx)," Train Loss:", str(loss.item()))

        #Save model parameters
        torch.save(network.state_dict(), '/results/model1.pth')
        torch.save(optimizer.state_dict(), '/results/optimizer1.pth')

        textfile = open("train_losses_1.txt", "w")
        for loss in train_losses:
            textfile.write(loss + "\n")
        textfile.close()

    def validation():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in zip(test_input, test_target):
                output = network(data)
                test_loss += nn.BCELoss(output, target,size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(test_input.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{}'.format(
            test_loss, correct, len(test_input.size()[0])))

    for epoch in range(1, n_epochs + 1):
        train(epoch)
        validation()
