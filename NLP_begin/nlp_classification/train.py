from numpy.lib.function_base import average
import torch
import os
import random
from torch import nn 
import numpy as np
from tqdm import tqdm
import transformers
from sklearn import metrics
from transformers.optimization import AdamW
from dataset import * #数据集
import config #超参数
from model import * #模型
from utils import *

import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore")

#from pytorch_pretrained.optimization import BertAdam #必须用这个 气死我了 艹

def train(epoch, train_loader, device, model, criterion, optimizer,scheduler,tensorboard_path):
    model.train()
    model = model.to(device)
    
    train_loss = 0.0
    total_batch = 0  # 记录进行到多少batch
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    
    train_loader = tqdm(train_loader)
    for i,(trains, labels)in enumerate(train_loader):  # 0是下标起始位置默认为0      (x, seq_len, mask), y    
        model.train()
        outputs = model(trains)
        optimizer.zero_grad()
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
             
        labels = labels.data.cpu().numpy()   
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()  #维度1是
        labels_all = np.append(labels_all, labels) #叠加
        predict_all = np.append(predict_all, predic) #叠加

        if i % 10 == 0: #s多久输出一次
            train_acc = metrics.accuracy_score(labels_all, predict_all)
            f1=metrics.f1_score(labels_all, predict_all, average='macro')
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'train_acc': '%.4f' % train_acc,"train_f1":"%.4f" % f1}
            train_loader.set_postfix(log=postfix)
   
        scheduler.step()
    train_loader.close()

def validate(epoch,validate_loader,device, model,criterion,tensorboard_path):
    validate_loss = 0.0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    model = model.to(device)
    model.eval()
    validate_loader = tqdm(validate_loader)
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        for i, (inputs, labels) in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
   
            outputs= model(inputs)
            loss = criterion(outputs, labels)
            validate_loss += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

            if i%10==0:
                acc = metrics.accuracy_score(labels_all, predict_all) #精确率
                f1=metrics.f1_score(labels_all, predict_all, average='macro')
                postfix = {'val_loss': '%.6f' % ( validate_loss/(i+1)), 'val_acc': '%.4f' % acc,"val_f1":"%.4f" % f1}
                validate_loader.set_postfix(log=postfix)

        precision=metrics.precision_score(labels_all, predict_all,average="macro")
        recall=metrics.recall_score(labels_all, predict_all,average="macro")
        acc = metrics.accuracy_score(labels_all, predict_all)
        f1=metrics.f1_score(labels_all, predict_all, average='macro')

        print("\n")
        print("Accuracy:%.6f  Precision:%.6f  Recall:%.6f  F1:%.6f"%(acc,precision,recall,f1))
    validate_loader.close()
    return f1

if __name__ == '__main__':

 #设置种子
    set_seed(1997)

    #加载数据集
    train_data, dev_data, _ = get_solve_dataset()
    train_iter = build_iterator(train_data)
    dev_iter = build_iterator(dev_data)
    #模型
    model = BERTRNN()    

    criterion = nn.CrossEntropyLoss() #交叉熵
    #判断是否要加载模型
    if config.load:
        model.load_state_dict(torch.load(config.model_save_path))
    
    #BERT的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer=AdamW(optimizer_grouped_parameters,lr=config.lr)
    scheduler=transformers.optimization.get_linear_schedule_with_warmup(optimizer,0.05*len(train_iter) * config.epoch,len(train_iter) * config.epoch)
    #训练
    val=0
    for epoch in range(config.epoch):
        #训练
        train(epoch, train_iter, config.device, model, criterion, optimizer,scheduler,config.tensorboard_path)
        val_f1=validate(epoch,dev_iter, config.device  ,model, criterion, config.tensorboard_path)
        if val_f1>val:
            torch.save(model.state_dict(), config.model_save_path)
            val=val_f1
    
    #保存模型 
    torch.save(model.state_dict(), config.model_save_path)


