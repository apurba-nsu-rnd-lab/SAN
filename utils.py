import os
import math
import random
import torch
import torchvision
from torchvision import transforms
import models, datasets

exp_dict = {
    1 : '5-split-mnist',
    2 : '20-split-cifar100',
    3 : '20-split-miniimagenet',
    4 : 'permuted-mnist',
    5 : 'sequence-of-5-datasets'
}


def generate_task_class_list(exp_id, n_cls, n_task, n_cls_per_task, verbose):
    # create task list and shuffle using random seed 1
    task = [x for x in range(n_cls)]
    if exp_id==4: task = [x for x in range(n_cls)]*2
    random.Random(1).shuffle(task)
    
    task_class_list = []
    c = 0
    
    for i in range(n_task):
        task_classes = []
        for j in range(n_cls_per_task):
            task_classes.append(task[c])
            c+=1
        task_class_list.append(task_classes)
    
    if verbose:
        print("task_class_list:\n", task_class_list)
    
    return task_class_list


def generate_split_data(dataset, task_class_list):
    split_datasets = [[] for i in range(len(task_class_list))]

    for data in dataset:
        for i in range(len(task_class_list)):
            if data[1] in task_class_list[i]:
                split_datasets[i].append(data)

    return split_datasets


def get_task_class_list(exp_id):
    if exp_id == 1:
        task_class_list = generate_task_class_list(exp_id, n_cls=10, n_task=5, n_cls_per_task=2, verbose=True)
    elif exp_id == 2 or exp_id == 3:
        task_class_list = generate_task_class_list(exp_id, n_cls=100, n_task=20, n_cls_per_task=5, verbose=True)
    elif exp_id == 4:
        task_class_list = generate_task_class_list(exp_id, n_cls=10, n_task=10, n_cls_per_task=2, verbose=True)
    elif exp_id == 5:
        task_class_list = [[x for x in range(10)]]*5
    else:
        raise Exception('Invalid Experiment ID.')
    
    return task_class_list


def get_transform(exp_id):
    # 5-split-MNIST
    if exp_id == 1 or exp_id == 4:
        transform = transforms.Compose(
            [
                transforms.Grayscale(1),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])

    elif exp_id == 2:
        transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

    elif exp_id == 3:
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])

    elif exp_id == 5:
        transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    else:
        raise Exception('Invalid Experiment ID.')

    return transform


def split_to_train_val(trainset, val_ratio=0.15):
    split = val_ratio 
    train_split, valid_split = torch.utils.data.random_split(
        trainset, [math.ceil(len(trainset)*(1-split)), math.floor(len(trainset)*split)]
    )
    return train_split, valid_split


def get_train_and_validation_set(root, val_ratio, exp_id, transform):
    if exp_id == 1 or exp_id == 4:
        trainset = torchvision.datasets.MNIST(root=root,train=True, download=True, transform=transform)
        train_split, valid_split = split_to_train_val(trainset)
    elif exp_id == 2:
        trainset = torchvision.datasets.CIFAR100(root=root,train=True, download=True, transform=transform)
        train_split, valid_split = split_to_train_val(trainset)
    elif exp_id == 3:
        miniimagenet_path = os.path.join(root, 'mini-imagenet/')
        trainset = datasets.MiniImagenetDataset(root=miniimagenet_path, mode='train')
        train_split, valid_split = split_to_train_val(trainset)
    elif exp_id == 5:
        train_split, valid_split = seq_5_train_val_set(root=root, transform=transform) # returns list of train and valid splits
    else:
        raise Exception('Invalid Experiment ID.')

    return train_split, valid_split


def get_testset(root, exp_id, transform):
    if exp_id == 1 or exp_id == 4:
        testset = torchvision.datasets.MNIST(root=root,train=False, download=True, transform=transform)
    elif exp_id == 2:
        testset = torchvision.datasets.CIFAR100(root=root,train=False, download=True, transform=transform)
    elif exp_id == 3:
        testset = datasets.MiniImagenetDataset(root=os.path.join(root, 'mini-imagenet/'), mode='test')
    elif exp_id == 5:
        testset = seq_5_test_set(root=root, transform=transform)
    else:
        raise Exception('Invalid Experiment ID.')

    return testset


def get_model(exp_id):
    if exp_id == 1:
        model = models.EquivalentNetMNIST()
    elif exp_id == 2:
        model = models.TwentySplit_CIFAR100()
    elif exp_id == 3:
        model = models.TwentySplit_MiniImagenet()
    elif exp_id == 4:
        model = models.Permuted_MNIST()
    elif exp_id == 5:
        model = models.Sequence_of_Datasets()
    else:
        raise Exception('Invalid Experiment ID.')

    return model


def seq_5_train_val_set(root, transform):
    trainsets, train_splits, valid_splits = [], [], []

    # preparing datasets
    trainsets.append(torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform))
    trainsets.append(torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transform))
    trainsets.append(torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=transform))
    trainsets.append(torchvision.datasets.SVHN(root=root, split='train', download=True, transform=transform))
    trainsets.append(torchvision.datasets.ImageFolder(root=os.path.join(root, 'notMNIST/Train/'), transform=transform))

    for trainset in trainsets:
        train_split, valid_split = split_to_train_val(trainset)
        train_splits.append(train_split)
        valid_splits.append(valid_split)

    return train_splits, valid_splits


def seq_5_test_set(root, transform):
    trainsets = []

    # preparing testsets
    trainsets.append(torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform))
    trainsets.append(torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform))
    trainsets.append(torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=transform))
    trainsets.append(torchvision.datasets.SVHN(root=root, split='test', download=True, transform=transform))
    trainsets.append(torchvision.datasets.ImageFolder(root=os.path.join(root, 'notMNIST/Test/'), transform=transform))

    return trainsets