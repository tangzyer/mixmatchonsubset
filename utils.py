import torch.nn as nn
import torch
import numpy as np
import math
from sklearn.metrics import accuracy_score
import torch.utils.data as Data
import torch.utils.data 
from torchvision import datasets
from subset import subset
import torchvision.transforms as transforms
import cifar10 as dataset


batch_size = 128
augment_K = 4

transform_train = transforms.Compose([
        dataset.RandomPadandCrop(32),
        dataset.RandomFlip(),
        dataset.ToTensor(),
    ])

transform_val = transforms.Compose([
    dataset.ToTensor(),
])

class MyDataset(Data.Dataset):

    def __init__(self, data, targets, transform=None):
        super(MyDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        augments = []
        if self.transform is not None:
            for i in range(augment_K):
                augments.append(self.transform(img))
        return (img, augments), target

    def __len__(self):
        return len(self.data)

class MyDatasetWithAugmentation(Data.Dataset):
    def __init__(self, data, targets, transform=None, augment_K=2):
        super(MyDatasetWithAugmentation, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.K = augment_K
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        augments = []
        if self.transform is not None:
            for i in range(self.K):
                augments.append(self.transform(img))
        return augments, target

    def __len__(self):
        return len(self.data)

def load_data():
    sub = subset(10)

    train_data = datasets.CIFAR10(root='.', train=True, download=True)
    test_data = datasets.CIFAR10(root='.', train=False, download=True)

    x_train = train_data.data
    x_test = test_data.data
    y_train = train_data.targets
    y_test = test_data.targets

    X_train_true = np.zeros(shape=(50000, 3, 32, 32), dtype=float)
    X_test_true = np.zeros(shape=(10000, 3, 32, 32), dtype=float)

    # data preparation
    for index, sample in enumerate(x_train):
        X_train_true[index] = sample.reshape(3,32,32).astype(float) / 255
    for index, sample in enumerate(x_test):
        X_test_true[index] = sample.reshape(3,32,32).astype(float) / 255

    # split
    from sklearn.model_selection import train_test_split

    train_X, train_y = X_train_true, y_train

    #train_X, _, train_y, _ = train_test_split(X_train_true, y_train, test_size=0.05, random_state=42)
    test_X, validation_X, test_y, validation_y = train_test_split(X_test_true, y_test, test_size=0.2, random_state=42)

    print(len(validation_X))

    test_labels = []
    train_labels = []
    X_validation = []
    validation_labels = []
    true_labels = []
    X_train = []
    X_test = []


    for index, y in enumerate(train_y):
        #size_of_subset, subset_y, obfuscated_y = sub.index_to_stack_obfuscated(y)
        multi_hot_subset = sub.index_to_limited_subset(y)
        X_train.append(train_X[index])
        train_labels.append(multi_hot_subset)
        true_labels.append(y)

    for index, y in enumerate(test_y):
        X_test.append(test_X[index])
        test_labels.append(y)

    for index,y in enumerate(validation_y):
        X_validation.append(validation_X[index])
        validation_labels.append(y)

    
    from torch.autograd import Variable
    print(torch.utils.data.dataloader.__file__)


  
    input = np.array(X_train).astype(np.float32)
    label = np.array(train_labels).astype(np.int64)
    true_labels = np.array(true_labels).astype(np.long)
    torch_dataset = MyDataset(input, label, transform_train)
    #torch_dataset_withaug = MyDatasetWithAugmentation(input, label, transform_train, augment_K)
    watch_dataset = MyDataset(input, true_labels, transform_val)
    train_loader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)
    #train_loader_augmentation = Data.DataLoader(torch_dataset_withaug, batch_size=batch_size, shuffle=True)
    watch_loader = Data.DataLoader(watch_dataset, batch_size=batch_size, shuffle=False)


    input = np.array(X_validation).astype(np.float32)
    label = np.array(validation_labels).astype(np.long)
    torch_dataset = MyDataset(input, label, transform_val)
    valid_loader = Data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, watch_loader, valid_loader
