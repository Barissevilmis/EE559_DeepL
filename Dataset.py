import torch
from torch.utils.data import Dataset, DataLoader

'''
This class exists to make data access easier and more efficient
After dataset is generated, it is loaded to this class within preprocess_dataset(*args) function
'''


class DataSet(Dataset):

    def __init__(self, data, target, classes, req_grad=True):
        self.data = data
        self.target = target
        self.classes = classes

        self.data = self.data.float()
        self.target = self.target.long()
        self.classes = self.classes.long()

        if req_grad:
            self.data.requires_grad = True
            #self.target.requires_grad = True
            #self.classes.requires_grad = True
        else:
            self.data.requires_grad = False
            self.target.requires_grad = False
            self.classes.requires_grad = False

        self.device = self.to_device()

        self.data = self.data.to(self.device)
        self.target = self.target.to(self.device)
        self.classes = self.classes.to(self.device)

        # Auxiliary loss: Get label digits and use them with auxiliary network
        self.class1, self.class2 = self.classes.split(1, dim=1)

    def __len__(self):
        print("Input shape: ", self.data.shape)
        print("Target shape: ", self.target.shape)
        print("Classes shape: ", self.classes.shape)
        return self.data.shape[0]

    def __getitem__(self, ind):

        return {'input': self.data[ind], 'target': self.target[ind], 'class1': self.class1[ind], 'class2': self.class2[ind]}

    def to_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def get_data(self):
        return self.data

    def get_target(self):
        return self.target

    def get_classes(self):
        return self.classes
