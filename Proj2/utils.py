import torch
import math


def generate_set(sample=1000):
    '''
    Generate the train and test dataset. Center of circle=(0.5,0.5), Radius=1/math.sqrt(2*math.pi)
    '''
#torch.empty(size=(sample, 2), dtype=float)
    train_data = torch.empty(sample, 2).uniform_(0, 1)
    test_data = train_data.sub(0.5).pow(2).sum(1) < 1/(2*math.pi)
    test_data = test_data.int()
    return train_data, test_data


if __name__ == "__main__":
    train_data, test_data = generate_set()
    print(train_data[test_data == 1].shape)
