import copy
import numpy as np
import torch.nn as nn

from models.mnistNet import MnistNet
from models.resnet_cifar import ResNet50_cifar10, Net_cifar, ALexNet
from models.resnet_tinyimagenet import resnet18


def create_model(args):
    dataset_name = args['dataset']

    if dataset_name == 'MNIST' or dataset_name == 'FASHION':
        net = MnistNet()

    elif dataset_name == "CIFAR":
        net = None
        if args['model_name'].lower() == 'cnn':
            net = Net_cifar()
        elif args["model_name"].lower() == "alexnet":
            net = ALexNet()
        elif args["model_name"].lower() == "resnet50":
            net = ResNet50_cifar10()

    elif dataset_name == "TINY":
        net = resnet18()

    return net


def poison_batch(X_, y_, poison_info, dataset_name):
    '''
        poisoning data
    '''
    if len(X_) != len(y_):
        print("WRONG in poison_batch!")
        return
    X = copy.deepcopy(X_)
    y = copy.deepcopy(y_)

    batch_size = len(X)
    poison_rate = poison_info['poison_rate']
    pattern = poison_info['pattern']
    target = poison_info['target']
    poison_index = list(
        np.random.choice(
            range(batch_size),
            int(poison_rate * batch_size),
            replace=False
        )
    )
    poison_index.sort()

    if dataset_name == 'MNIST' or dataset_name == 'FASHION':
        for i in poison_index:
            for p in pattern:
                X[i][0][p[0]][p[1]] = 1
            y[i] = target
    elif dataset_name == 'CIFAR' or dataset_name == 'TINY':
        for i in poison_index:
            for p in pattern:
                X[i][0][p[0]][p[1]] = 1
                X[i][1][p[0]][p[1]] = 1
                X[i][2][p[0]][p[1]] = 1
            y[i] = target
    return X, y


def local_train(net, train_iter, optimizer, device, num_epochs, dataset_name, poison_info=None,
                scheduler=None):
    net = net.to(device)
    net.train()
    for epoch in range(num_epochs):
        for X, y in train_iter:
            if poison_info is not None:  # 毒化
                X, y = poison_batch(X, y, poison_info, dataset_name)
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = net(X)
            l = nn.functional.cross_entropy(y_hat, y)
            l.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
