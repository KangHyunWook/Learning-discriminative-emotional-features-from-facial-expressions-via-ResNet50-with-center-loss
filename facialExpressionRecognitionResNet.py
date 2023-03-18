from collections import namedtuple
from config import get_config

import os
import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride=1, padding =1, bias= False)

        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace = True)

        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride = stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):
        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3,
                               stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size = 1,
                               stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.relu = nn.ReLU(inplace = True)

        if downsample:
            conv = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size = 1,
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(self.expansion * out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None

        self.downsample = downsample

    def forward(self, x):

        i = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            i = self.downsample(i)

        x += i
        x = self.relu(x)

        return x


ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])

resnet18_config = ResNetConfig(block = BasicBlock,
                                n_blocks = [2,2,2,2],
                                channels = [64, 128, 256, 512])



resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])



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


#code adapted from: https://github.com/bentrevett/pytorch-image-classification/blob/master/5_resnet.ipynb
class ResNet(nn.Module):
    def __init__(self, config, output_dim):
        super().__init__()

        block, n_blocks, channels = config
        self.in_channels = channels[0]

        assert len(n_blocks) == len(channels) == 4

        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, stride = 2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride= 2,padding=1)

        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride=2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride=2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(self.in_channels, output_dim)

    def get_resnet_layer(self, block, n_blocks, channels, stride=1):
        layers = []

        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False

        layers.append(block(self.in_channels, channels, stride, downsample))

        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)

        return h, x

train_config = get_config(mode='train')
test_config = get_config(mode='test')

train_dir = os.path.join(train_config.data_folder, train_config.mode)
test_dir = os.path.join(test_config.data_folder, test_config.mode)


'''
calculate means and stds for normalization.
'''

# train_data = datasets.ImageFolder(root=train_dir,
#                                 transform = transforms.ToTensor())
# means = torch.zeros(3)
# stds = torch.zeros(3)
#
# for img, label in train_data:
#     means += torch.mean(img, dim=(1,2))
#     stds += torch.std(img, dim=(1,2))
#
# means /= len(train_data)
# stds /= len(train_data)
#
# print('means:', means)
# print('stds:', stds)

pretrained_size = 224

pretrained_means = [0.575, 0.450, 0.401]
pretrained_stds = [0.209, 0.191, 0.183]


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


SEED = 336
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

# print('output_dim:', OUTPUT_DIM)
# exit()
OUTPUT_DIM = len(test_data.classes)

model = ResNet(resnet50_config, OUTPUT_DIM)

pretrained_model=models.resnet50(pretrained=True)

pretrained_model.fc=nn.Linear(2048, OUTPUT_DIM)

model.load_state_dict(pretrained_model.state_dict())

model = nn.DataParallel(model).cuda()

EPOCHS = 50
BATCH_SIZE=64

n = len(train_data)  # total number of examples
n_dev = int(0.3 * n)

from sklearn.utils import shuffle
from torch import optim

n_list=shuffle(np.arange(n))

dev_data = torch.utils.data.Subset(train_data, n_list[:n_dev])  # take first 10%
train_data = torch.utils.data.Subset(train_data, n_list[n_dev:])  # take the rest

trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)
devloader = DataLoader(dev_data, batch_size = BATCH_SIZE, shuffle = False)
testloader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

saved_model_name = 'model.ckpt'
curr_patience = patience = 6
best_valid_loss = float('inf')

num_trials=1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

optimizer_model = train_config.optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=train_config.lr_model)

criterion = nn.CrossEntropyLoss(reduction='mean')

criterion_cent = CenterLoss(num_classes=OUTPUT_DIM, feat_dim=2048)

optimizer_centloss = optim.Adam(criterion_cent.parameters(), lr = train_config.lr_cent)

for epoch in range(EPOCHS):
    model.train()
    total_step = len(trainloader)

    for batch, (data, labels) in enumerate(trainloader):
        data = data.to(device)
        labels = labels.to(device)

        features, outputs = model(data)
        loss_xent = criterion(outputs, labels)

        if train_config.center:
            loss_cent = criterion_cent(features, labels)
            loss = loss_xent + loss_cent*train_config.cent_weight
        else:
            loss = loss_xent

        optimizer_model.zero_grad()
        if train_config.center:
            optimizer_centloss.zero_grad()

        loss.backward()
        optimizer_model.step()

        if train_config.center:
            for param in criterion_cent.parameters():
                param.grad.data *= (1.8 / train_config.cent_weight)
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

        epoch_test_acc*=100
        f=open('results.csv','w')
        f.write('test_loss\ttest_acc\n')
        f.write(str(np.round(epoch_test_loss.detach().cpu().numpy(),2))+'\t'+str(np.round(epoch_test_acc,3)))
        f.close()

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

epoch_test_acc*=100
print('test_loss: {:.3f} | test_acc: {:.3f}'.format(epoch_test_loss, epoch_test_acc))


f=open('results.csv','w')
f.write('test_loss\ttest_acc\n')
f.write(str(np.round(epoch_test_loss.detach().cpu().numpy(),2))+'\t'+str(np.round(epoch_test_acc,3)))
f.close()
























#
