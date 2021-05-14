import utils
import torch

best_train_loss_fc = torch.load("best_train_loss_fc.pt")
best_train_acc_fc = torch.load("best_train_acc_fc.pt")
best_val_acc_fc = torch.load("best_val_acc_fc.pt")

utils.plot_train_test(best_train_loss_fc,
                      best_train_acc_fc, best_val_acc_fc, "FC")

best_train_loss_cnn = torch.load("best_train_loss_cnn.pt")
best_train_acc_cnn = torch.load("best_train_acc_cnn.pt")
best_val_acc_cnn = torch.load("best_val_acc_cnn.pt")

utils.plot_train_test(best_train_loss_cnn,
                      best_train_acc_cnn, best_val_acc_cnn, "CNN")

best_train_loss_auxn = torch.load("best_train_loss_auxn.pt")
best_train_acc_auxn = torch.load("best_train_acc_auxn.pt")
best_val_acc_auxn = torch.load("best_val_acc_auxn.pt")

utils.plot_train_test(best_train_loss_auxn,
                      best_train_acc_auxn, best_val_acc_auxn, "AUXN")

utils.plot_val_scores()
