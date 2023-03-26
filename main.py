from config import get_config
from solver import Solver

from sklearn.utils import shuffle
from torch import optim

import models

import torch.nn as nn
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models import *

def getPathList(root):
    items = os.listdir(root)
    pathList=[]
    for item in items:
        full_path = os.path.join(root, item)
        if os.path.isfile(full_path):
            pathList.append(full_path)
        else:
            pathList.extend(getPathList(full_path))

    return pathList

'''
calculate statistics of the data.
'''

# pathList=getPathList(os.path.join(train_config.data_folder, 'test'))
#
# print(len(pathList))
# emo_count_map={}
# for path in pathList:
#     emo=path.split(os.path.sep)[-2]
#     if emo not in emo_count_map:
#         emo_count_map[emo]=1
#     else:
#         emo_count_map[emo]+=1
#
# print(emo_count_map)
# exit()


if __name__=='__main__':

    SEED = 336
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    pretrained_size = 224

    pretrained_means = [0.575, 0.450, 0.401]
    pretrained_stds = [0.209, 0.191, 0.183]

    train_config = get_config(mode='train')
    dev_config = get_config(mode='dev')
    test_config = get_config(mode='test')

    DATA_NAME=train_config.data_folder.split(os.path.sep)[-1]

    train_dir = os.path.join(train_config.data_folder, train_config.mode)
    test_dir = os.path.join(test_config.data_folder, test_config.mode)

    train_config.save_weights_name=DATA_NAME+'-'+str(train_config.center)+'-'+'model.ckpt'

    train_transforms = transforms.Compose([
                                    transforms.Resize(pretrained_size),
                                    transforms.RandomRotation(5),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.RandomCrop(pretrained_size, padding=10),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = pretrained_means, std = pretrained_stds)
                                ])

    test_transforms = transforms.Compose([
                                    transforms.Resize(pretrained_size),
                                    transforms.CenterCrop(pretrained_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean = pretrained_means,
                                                         std = pretrained_stds)
                            ])

    train_data = datasets.ImageFolder(root=train_dir, transform = train_transforms)

    test_data = datasets.ImageFolder(root = test_dir, transform = test_transforms)

    n = len(train_data)  # total number of examples
    n_dev = int(0.3 * n)

    n_list=shuffle(np.arange(n))

    dev_data = torch.utils.data.Subset(train_data, n_list[:n_dev])  # take first 30%
    train_data = torch.utils.data.Subset(train_data, n_list[n_dev:])  # take the rest

    trainloader = DataLoader(train_data, batch_size=train_config.batch_size, shuffle=False)
    devloader = DataLoader(dev_data, batch_size = dev_config.batch_size, shuffle = False)
    testloader = DataLoader(test_data, batch_size = test_config.batch_size, shuffle = False)

    solver = Solver
    solver = solver(train_config, dev_config, test_config, trainloader, devloader, testloader, is_train=True)
    solver.build()

    if train_config.vis:
        solver.visualize()
    elif train_config.run_type=='train':
        solver.train()
    elif train_config.run_type=='test':
        epoch_test_loss, epoch_test_acc = solver.evaluate(testloader)
        print('test_loss: {:.3f} | test_acc: {:.3f}'.format(epoch_test_loss, epoch_test_acc))
