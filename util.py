import os
import gzip
import numpy as np
from torchvision import datasets,transforms

class DealDataset():
    def __init__(self, folder, data_name, label_name,transform=None):
        (train_set, train_labels) = load_data(folder, data_name, label_name) 
        self.train_set = train_set
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_set[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_set)

def load_data(data_folder, data_name, label_name):
    with gzip.open(os.path.join(data_folder,label_name), 'rb') as lbpath:
        y_data = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(os.path.join(data_folder,data_name), 'rb') as imgpath:
        x_data = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_data), 28, 28)
    return (x_data, y_data)

def get_data():
    # load local mnist
    folder = '/data1/ywh/MNIST_data'
    train_data = 'train-images-idx3-ubyte.gz' 
    train_label = 'train-labels-idx1-ubyte.gz'
    test_data = 't10k-images-idx3-ubyte.gz'
    test_label = 't10k-labels-idx1-ubyte.gz'
    # dealdataset
    trainDataset = DealDataset(folder, train_data, train_label,transform=transforms.ToTensor())
    testDataset = DealDataset(folder, test_data, test_label,transform=transforms.ToTensor())

    return trainDataset, testDataset