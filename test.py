import FC
import torch
import torch.nn as nn
import utils


model = FC.DenseNet()
criterion = nn.CrossEntropyLoss()
train_avg, train_acc_avg, test_acc_avg = utils.avg_scores(model, criterion, epochs=100, sample_size=1000,
                                                          lr=1e-1, weight_decay=1e-3, batch_size=100, aux_param=.2)
print(train_avg, train_acc_avg, test_acc_avg)
