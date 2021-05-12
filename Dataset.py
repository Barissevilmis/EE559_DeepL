import torch
from torch.utils.data import Dataset, DataLoader

'''
This class exists to make data access easier and more efficient
After dataset is generated, it is loaded to this class within preprocess_dataset(*args) function
'''
class DataSet(Dataset):

    def __init__(self, data, target, classes, req_grad = True):
        self.data = data
        self.target = target
        self.classes = classes

        self.data = self.data.float()
        self.target  = self.target.float()
        self.classes = self.classes.float()

        if req_grad:
            self.data.requires_grad = True
            self.target.requires_grad = True
            self.classes.requires_grad = True
        else:
            self.data.requires_grad = False
            self.target.requires_grad = False
            self.classes.requires_grad = False

        self.device = self.to_device()

        self.data = self.data.to(self.device)
        self.target = self.target.to(self.device)
        self.classes = self.classes.to(self.device)

    def __len__(self):
        print("Input shape: ", self.data.shape)
        print("Target shape: ", self.target.shape)
        print("Classes shape: ", self.classes.shape)
        return self.data.shape[0]

    def __getitem__(self, ind):

        return {'input': self.data[ind], 'target': self.target[ind], 'classes':self.classes[ind]}

    def to_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
