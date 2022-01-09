# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 20:20:16 2021

@author: ZHUANG
"""
import torch
import torch.nn as nn
import torch.nn.functional
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from mmd import mmd_rbf
import net1 as models
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,recall_score,precision_score

lr=0.00001
batch_size = 100   
epochs = 500
hidden_size = 500
class_num = 5

class MyDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path)
        self.label = np.load(label_path)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)

def load_training(): 
    data_path = 'E:/cross_corpus_database/B-C/IS10_feature/CASIA_IS10_1000_neutral.npy' 
    label_path = 'E:/cross_corpus_database/B-C/label_CASIA_neutral.npy' 
    dataset = MyDataset(data_path,label_path) 
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)  
    return train_loader

def load_testing():
    data_path = 'E:/cross_corpus_database/B-C/IS10_feature/Berlin_IS10_408_neutral.npy'
    label_path = 'E:/cross_corpus_database/B-C/label_Berlin_neutral.npy'
    dataset = MyDataset(data_path,label_path)
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)  
    return test_loader

source_loader = load_training()
target_loader = load_testing()

len_target_dataset = len(target_loader.dataset)
len_target_loader = len(target_loader) 
len_source_loader = len(source_loader)

D_M=0
D_C1=0
D_C2=0
D_CLS=0
 
def train(epoch, model):
    optimizer = torch.optim.Adam([
            {'params': model.feature_layers.parameters()},
            {'params': model.DNN.parameters(),'lr': lr},
            {'params': model.cls_fc.parameters(),'lr': lr},
            ], lr=lr, weight_decay=5e-4)

    Loss1 = nn.BCELoss() 
    Loss2 = nn.BCELoss()
    
    global D_M, D_L, D_C1, D_C2, D_CLS
    d_m = 0
    d_c1 = 0
    d_c2 = 0
    d_cls = 0
      
    model.train()    
    iter_source = iter(source_loader) 
    iter_target = iter(target_loader)
        
    num_iter = len_source_loader
    #num_iter = len_target_loader
       
    if D_M==0 and D_C1==0 and D_C2==0 and D_CLS==0:
        A = 1
        B = 1
        C = 1
        D = 1
    else:             
        D_C1 = D_C1/num_iter  
        D_C2 = D_C2/num_iter   
        D_CLS = D_CLS/num_iter  
        D_M = D_M/num_iter   
        
        A = 0.1*D_C1/(D_M + D_L + D_C1 + D_C2 + D_CLS)
        B = 0.1*D_C2/(D_M + D_L + D_C1 + D_C2 + D_CLS)
        C = 5*D_CLS/(D_M + D_L + D_C1 + D_C2 + D_CLS)
        D = 2*D_M/(D_M + D_L + D_C1 + D_C2 + D_CLS)
                 
    train_loss = 0
    for i in range(0, num_iter):       
        data_source, label_source = iter_source.next()
        data_target, label_target = iter_target.next() 
        s_label = label_source
        label_source = label_source.clone().detach().long()
        
        min_max_scaler = MinMaxScaler()  
        data_source = min_max_scaler.fit_transform(data_source)
        data_target = min_max_scaler.fit_transform(data_target)
        data_source = torch.Tensor(data_source)  
        data_target = torch.Tensor(data_target)          
        
        if i % len_target_loader == 0:
            iter_target = iter(target_loader)                                   
        
        if i % len_source_loader == 0:
            iter_source = iter(source_loader)
            
        noise = torch.Tensor(np.random.normal(loc=0.0, scale=1.0, size=(1582,)))        
        data_noise_source = data_source + 0.6*noise
        data_noise_target = data_target + 0.6*noise
        
        if torch.cuda.is_available():           
            data_noise_source = data_noise_source.cuda()
            data_noise_target = data_noise_target.cuda()
            data_source = data_source.cuda()
            data_target = data_target.cuda()
            label_source = label_source.cuda()
        data_noise_source = Variable(data_noise_source)
        data_noise_target = Variable(data_noise_target)
        data_source = Variable(data_source)
        label_source = Variable(label_source)
        data_target = Variable(data_target)       
        
        optimizer.zero_grad()                 
        out = model(data_noise_source, data_noise_target)
        source_decoder_out, target_decoder_out = out[1],out[5]
        s_dnn_out, t_dnn_out, source_p, target_p = out[2], out[6], out[3], out[7] 
                                            
        pred_label = torch.nn.functional.log_softmax(source_p, dim=1)
        loss_cls = torch.nn.functional.nll_loss(pred_label, label_source)   
        t_label = torch.nn.functional.softmax(target_p, dim=1)            
        
        loss_mmd = mmd_rbf(s_dnn_out, t_dnn_out)
        loss1 = Loss1(source_decoder_out, data_source)
        loss2 = Loss2(target_decoder_out, data_target)
        
        d_m = loss_mmd.cpu().item()
        d_c1 = loss1.cpu().item()
        d_c2 = loss2.cpu().item()
        d_cls = loss_cls.cpu().item()
     
        loss = A*loss1 + B*loss2 + C*loss_cls + D*loss_mmd 
        
        train_loss += loss
                
        loss.backward()
        optimizer.step()
                
        D_M += d_m
        D_C1 += d_c1
        D_C2 += d_c2
        D_CLS += d_cls
        
    train_loss /= num_iter
    print('Train Epoch: {}\ttrain loss: {:.2f}'.format(epoch, train_loss.item()))   
  
def test(model):
    model.eval()
    with torch.no_grad():
        x_test = np.load('E:/cross_corpus_database/B-C/IS10_feature/Berlin_IS10_408_neutral.npy')
        y_test = np.load('E:/cross_corpus_database/B-C/label_Berlin_neutral.npy')
  
        min_max_scaler = MinMaxScaler()   
        x_test = min_max_scaler.fit_transform(x_test)
        x_test = torch.Tensor(x_test).cuda()
        y_test = torch.Tensor(y_test).cuda().long()    
        x_test = Variable(x_test)
        y_test = Variable(y_test) 
        
        out = model(x_test, x_test)
        s_output= out[3]        
      
        test_loss = F.nll_loss(F.log_softmax(s_output, dim = 1), y_test).item()
        pred = s_output.data.max(1)[1]      # get the index of the max log-probability   
        correct = pred.eq(y_test.data.view_as(pred)).cpu().sum().item()
        
        y_pred = pred.cpu().numpy()
        y_true = y_test.detach().cpu().numpy()
        
        recall = recall_score(y_true, y_pred, average='macro') 
        w_recall = recall_score(y_true, y_pred, average='weighted')
        
    print('test loss: {:.2f}, Accuracy: {}/{} ({:.2f}%)\t'.format(
            test_loss, correct, len_target_dataset,
            100. * correct / len_target_dataset))
  
    return correct, recall, w_recall
            
if __name__ == '__main__':
    model = models.my_net(num_classes=class_num,hidden_size=hidden_size)
    correct = 0
    recall = 0
    w_recall = 0
    print(model)
    if torch.cuda.is_available():
        model.cuda()

    for epoch in range(1, epochs + 1):
        train(epoch, model)
        t_correct, t_recall, t_w_recall= test(model)
        if t_correct > correct:
            correct = t_correct
        if t_recall > recall:
            recall = t_recall
        if t_w_recall > w_recall:
            w_recall = t_w_recall
            #torch.save(model, 'model.pkl')
        print('max accuracy{: .2f}%'.format(100. * correct / len_target_dataset ))    
        print('max recall{: .2f}%'.format(100. * recall)) 
        print('max w_recall{: .2f}%\n'.format(100. * w_recall))


