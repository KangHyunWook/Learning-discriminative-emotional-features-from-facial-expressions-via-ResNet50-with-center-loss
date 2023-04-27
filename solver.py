from models import *
from sklearn.metrics import f1_score

from torch.utils.data import DataLoader

from sklearn.decomposition import PCA

import torch
import torchvision
import models

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#code adapted from: https://github.com/KaiyangZhou/pytorch-center-loss
class CenterLoss(nn.Module):
    def __init__(self, num_classes=7, feat_dim=2048):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

    def forward(self, x, labels):

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes,
                                                                                    batch_size).t()
#         distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)

        distmat = 1*distmat + -2*torch.dot(x,self.centers.t())

        classes = torch.arange(self.num_classes).long()
        classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

#code adapted from: https://github.com/declare-lab/MISA/blob/master/src/solver.py
class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):
        self.train_config = train_config
        self.test_config = test_config
        self.train_data_loader= train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train=is_train
        self.model=model

    def build(self, cuda=True):
        if self.model is None:
            pretrained_model=torchvision.models.resnet50(pretrained=True)

            IN_FEATURES=pretrained_model.fc.in_features

            self.model = getattr(models, 'ResNet')(resnet50_config, self.train_config.n_classes)
            pretrained_model.fc=nn.Linear(IN_FEATURES, self.train_config.n_classes)
            self.model.load_state_dict(pretrained_model.state_dict())


        if torch.cuda.is_available() and cuda:
            self.model = nn.DataParallel(self.model).cuda()
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if self.is_train:
            self.optimizer_model = self.train_config.optimizer(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.train_config.lr_model)

        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        self.criterion_cent = CenterLoss(num_classes=self.train_config.n_classes, feat_dim=2048)

        self.optimizer_centloss = self.train_config.optimizer(self.criterion_cent.parameters(), lr = self.train_config.lr_cent)

    def saveResults(self, config, loss, acc):
        f=open('results.csv', config.file_mode)
        if config.file_mode=='w':
            f.write('alpha\ttest_loss\ttest_acc\n')

        f.write(str(self.train_config.alpha)+'\t'+str(np.round(loss.detach().cpu().numpy(),2))+'\t'+str(np.round(acc,3)))
        f.write('\n')
        f.close()

    def train(self):
        num_trials=1

        curr_patience = patience = self.train_config.patience
        best_valid_loss = float('inf')

        for epoch in range(self.train_config.n_epochs):
            for batch, (data, labels) in enumerate(self.train_data_loader):
                self.model.train()
                data = data.to(self.device)
                labels = labels.to(self.device)

                features, outputs = self.model(data)
                loss_xent = self.criterion(outputs, labels)

                if self.train_config.center:
                    loss_cent = self.criterion_cent(features, labels)
                    loss = loss_xent + loss_cent*self.train_config.cent_weight
                else:
                    loss = loss_xent

                self.optimizer_model.zero_grad()
                if self.train_config.center:
                    self.optimizer_centloss.zero_grad()

                loss.backward()
                self.optimizer_model.step()

                if self.train_config.center:
                    for param in self.criterion_cent.parameters():
                        param.grad.data = param.grad.data*(self.train_config.alpha/self.train_config.cent_weight)
                    self.optimizer_centloss.step()

                # torch.nn.utils.clip_grad_value_([param for param in model.parameters()
                                                        # if param.requires_grad], 1.0)

            epoch_val_loss, epoch_val_acc, f1 = self.evaluate(self.dev_data_loader, is_load=False)

            print('val_loss: {:.3f} | acc: {:.3f} | F1: {:.3f}'.format(epoch_val_loss, epoch_val_acc, f1))
            if epoch_val_loss<best_valid_loss:
                torch.save(self.model.state_dict(), self.train_config.save_weights_name)
                torch.save(self.optimizer_centloss.state_dict(), 'optim_best.std')
                best_valid_loss=epoch_val_loss
                curr_patience = patience
            else:
                curr_patience-=1
                if curr_patience<=-1:
                    self.model.load_state_dict(torch.load(self.train_config.save_weights_name))
                    self.optimizer_centloss.load_state_dict(torch.load('optim_best.std'))
                    num_trials-=1

            if num_trials<=0:
                print('Running out of patience, training finished')

                epoch_test_loss, epoch_test_acc, f1 = self.evaluate(self.test_data_loader)
                print('test_loss: {:.3f} | test_acc: {:.3f} f1: {:.3f}'.format(epoch_test_loss, epoch_test_acc, f1))

                self.saveResults(self.test_config, epoch_test_loss, epoch_test_acc)
                exit()

        epoch_test_loss, epoch_test_acc, f1 = self.evaluate(self.test_data_loader)
        print('test_loss: {:.3f} | test_acc: {:.3f} f1: {:.3f}'.format(epoch_test_loss, epoch_test_acc, f1))
        self.saveResults(self.test_config, epoch_test_loss, epoch_test_acc)

    def evaluate(self, dataloader, is_load=True):

        if is_load:
            self.model.load_state_dict(torch.load(self.train_config.save_weights_name))

        self.model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            epoch_acc=0.0

            correct=0
            total=0

            pred_list=[]
            true_list=[]
            for data, labels in dataloader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                _, outputs= self.model(data)

                preds = torch.argmax(outputs, dim=1)

                pred_list.extend(list(preds.detach().cpu().numpy()))
                true_list.extend(list(labels.detach().cpu().numpy()))

                correct += (preds==labels).sum().item()
                total+=data.shape[0]

                loss = self.criterion(outputs, labels)
                epoch_loss += loss

            epoch_loss/=len(dataloader)
            epoch_acc= correct/total
            epoch_f1 = f1_score(true_list, pred_list, average='weighted')

        epoch_acc*=100

        if self.train_config.run_type=='test' and self.train_config.confusion:

            columns=['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            conf_mat =  confusion_matrix(true_list, pred_list)

            if 'RAF-DB' in self.train_config.data_folder:
                columns=['surprise', 'fear', 'disgust', 'happy', 'sad', 'angry', 'neutral']

            row_wise_sum= np.sum(conf_mat, axis=1, keepdims=True, dtype='float64')
            row_wise_sum = np.broadcast_to(row_wise_sum, (7,7))

            conf_mat = conf_mat / row_wise_sum
            conf_mat=np.round(conf_mat, 2)

            df_cm = pd.DataFrame(conf_mat, index = columns,
                              columns = columns)
            plt.figure(figsize = (10,7))
            sn.heatmap(df_cm, annot=True, cmap='Blues')

            plt.show()
            exit()

        return epoch_loss, epoch_acc, epoch_f1*100

    def visualize(self):

        self.model.load_state_dict(torch.load(self.train_config.save_weights_name))

        colors={0: 'k', 1:'g', 2: 'y', 3:'b', 4: 'c', 5: 'r', 6: 'm'}
        colors_flag={0:0,1:0,2:0,3:0,4:0,5:0,6:0}
        int_label2str={0:'surprise', 1: 'fear', 2:'disgust', 3:'happy', 4:'sad', 5: 'anger', 6:'neutral'}

        pca = PCA(n_components=2)
        deep_feats_list=[]
        labels_list=[]

        for features, labels in self.train_data_loader:
            features.to(self.device)

            for i in range(features.shape[0]):
                x=torch.unsqueeze(features[i], dim=0)
                deep_feats, preds = self.model(x)

                deep_feats_list.append(deep_feats[0].detach().cpu().numpy())
                labels_list.append(labels[i].item())

        pcaed_deep_feats=pca.fit_transform(deep_feats_list)

        for i in range(pcaed_deep_feats.shape[0]):
            key_label=int(labels_list[i])
            if colors_flag[key_label]==0:
                plt.scatter(pcaed_deep_feats[i,0], pcaed_deep_feats[i,1], marker='o', c=colors[key_label], label=int_label2str[key_label])
                colors_flag[key_label]=1
            else:
                plt.scatter(pcaed_deep_feats[i,0], pcaed_deep_feats[i,1], marker='o', c=colors[key_label])

        plt.legend(prop={'size': 10}, loc='upper right')
        plt.show()
        print(pcaed_deep_feats.shape)
