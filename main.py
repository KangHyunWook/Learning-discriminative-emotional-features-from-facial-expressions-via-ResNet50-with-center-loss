from sklearn.utils import shuffle
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader

import torch
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn

import datasets

#code adapted from: https://github.com/KaiyangZhou/pytorch-center-loss
class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels):
      
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, 
                                                                                    batch_size).t()
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        
        self.fc1 = nn.Linear(128*6*6, 256)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 128*6*6)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)

        return x, y

parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist'])
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=1e-3, help="learning rate for center loss")

parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--model', type=str, default='cnn')
parser.add_argument('--center', action='store_true', help='activate center loss')

args = parser.parse_args()

emotion_categories=os.listdir(r'C:\data\images\validation')

import cv2

class CusDataset(torch.utils.data.Dataset):
    def __init__(self, file_list):
 
        self.img_list=[]
        self.label_list=[]

        for file in file_list:
            self.img_list.append(cv2.imread(file)/255.0)
        for file_path in file_list:
            label = file_path.split(os.path.sep)
            label=emotion_categories.index(label[-2])
            self.label_list.append(label)
                
    def __getitem__(self, index):        

        feature = torch.from_numpy(self.img_list[index]).float()
        feature = feature.permute(2,0,1)
        label = torch.from_numpy(np.asarray(self.label_list[index])).long()
        
        return feature, label
        
    def __len__(self):
        return len(self.img_list)

def getFileList(root):
    items = os.listdir(root)
    fileList=[]
    for item in items:
        full_path = os.path.join(root, item)
        
        if os.path.isfile(full_path):
            fileList.append(full_path)
        else:
            fileList.extend(getFileList(full_path))
            
    return fileList

num_trials=1

if __name__ == '__main__':

    SEED = 336
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    criterion = nn.CrossEntropyLoss(reduction='mean')
    model = ConvNet(num_classes=7)

    #read datasets
    trainfileList = getFileList(r'C:\data\images\train')
    trainfileList = shuffle(trainfileList)
    
    n_train = len(trainfileList)
    n_dev=int(n_train*0.2)

    devfileList=trainfileList[:n_dev]
    trainfileList=trainfileList[n_dev:]
    testfileList = getFileList(r'C:\data\images\validation')
    
    # trainloader, testloader = dataset.trainloader, dataset.testloader
    train_data = CusDataset(trainfileList)
    dev_data = CusDataset(devfileList)
    test_data = CusDataset(testfileList)
    BATCH_SIZE=64
    trainloader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle=True)
    devloader = DataLoader(dev_data, batch_size = BATCH_SIZE, shuffle=False)
    testloader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle=False)
    
    saved_model_name = 'model.ckpt'
    curr_patience = patience = 6
    best_valid_loss = float('inf')

    model = nn.DataParallel(model).cuda()

    criterion_cent = CenterLoss(num_classes=7, feat_dim=256)

    optimizer_model = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = 1e-3)
    optimizer_centloss = optim.Adam(criterion_cent.parameters(), lr = args.lr_cent)

    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cent_weight = 0.6

    for epoch in range(30):
        model.train()
        total_step = len(trainloader)        
        print('=====Epoch {}====='.format(epoch+1))
        for batch_idx, (data, labels) in enumerate(trainloader):
            
            data = data.to(device)
            labels = labels.to(device)
            #todo: remove center loss with option
            features, outputs = model(data)
            loss_xent = criterion(outputs, labels)

            if args.center:
                loss_cent = criterion_cent(features, labels)
                loss = loss_xent + loss_cent*cent_weight
            else:
                loss=loss_xent
    
            optimizer_model.zero_grad()
            if args.center:
                optimizer_centloss.zero_grad()

            loss.backward()
            optimizer_model.step()            

            if args.center:
                for param in criterion_cent.parameters():
                    param.grad.data *= (1. / cent_weight)
                optimizer_centloss.step()
                
            # torch.nn.utils.clip_grad_value_([param for param in model.parameters() 
                                        # if param.requires_grad], 1.0)

        model.eval()
        
        with torch.no_grad():
            total = 0
            epoch_val_loss = 0.0
            epoch_val_acc=0.0
            
            for data, labels in devloader:
                data = data.to(device)
                labels = labels.to(device)
                _, outputs= model(data)
                
                preds = torch.argmax(outputs, dim=1)
                
                correct = (preds==labels).sum().item()
               
                acc = correct/data.shape[0]

                loss = criterion(outputs, labels)
                epoch_val_loss+=loss
                epoch_val_acc+=acc
               
            epoch_val_loss/=len(devloader)
            epoch_val_acc/=len(devloader)
        print('val_loss: {:.3f} | acc: {:.3f}'.format(epoch_val_loss, 100*acc))
        if epoch_val_loss<best_valid_loss:
            torch.save(model.state_dict(), 'model.ckpt')
            torch.save(optimizer_centloss.state_dict(), 'optim_best.std')
            best_valid_loss=epoch_val_loss
            curr_patience = patience
        else:
            curr_patience-=1
            if curr_patience<=-1:
                model.load_state_dict(torch.load('model.ckpt'))
                optimizer_centloss.load_state_dict(torch.load('optim_best.std'))
                num_trials-=1
        if num_trials<=0:
            print('Running out of patience, training finished')

            model.load_state_dict(torch.load('model.ckpt'))
                   
            model.eval()
            with torch.no_grad():
                total = 0
                epoch_test_loss = 0.0
                epoch_test_acc=0.0
                
                for data, labels in testloader:
                    data = data.to(device)
                    labels = labels.to(device)
                    _, outputs= model(data)
                    
                    preds = torch.argmax(outputs, dim=1)
                    
                    correct = (preds==labels).sum().item()
                   
                    acc = correct/data.shape[0]

                    loss = criterion(outputs, labels)
                    epoch_test_loss+=loss
                    epoch_test_acc+=acc
                   
                epoch_test_loss/=len(testloader)
                epoch_test_acc/=len(testloader)
            print('test_loss: {:.3f} | test_acc: {:.3f}'.format(epoch_test_loss, 100*epoch_test_acc))
            exit()

model.load_state_dict(torch.load('model.ckpt'))
       
model.eval()
with torch.no_grad():
    total = 0
    epoch_test_loss = 0.0
    epoch_test_acc=0.0
    
    for data, labels in testloader:
        data = data.to(device)
        labels = labels.to(device)
        _, outputs= model(data)
        
        preds = torch.argmax(outputs, dim=1)
        
        correct = (preds==labels).sum().item()
       
        acc = correct/data.shape[0]

        loss = criterion(outputs, labels)
        epoch_test_loss+=loss
        epoch_test_acc+=acc
       
    epoch_test_loss/=len(testloader)
    epoch_test_acc/=len(testloader)
print('test_loss: {:.3f} | test_acc: {:.3f}'.format(epoch_test_loss, 100*epoch_test_acc))





