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
from predictv2 import single_model_predict

def train(epoch, train_loader, device, model, criterion, optimizer,scheduler, pgd):
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
  
        loss = criterion(outputs,labels)
        loss.backward()

        if pgd is not None:
            pgd.backup_grad()
            K=3#小步走的次数 对抗的次数
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K-1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                loss_adv = criterion(model(trains),labels)
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore() # 恢复embedding参数

        optimizer.step()
      #  optimizer.zero_grad()
        model.zero_grad() #应该是与上述等效的
 
        labels = labels.data.cpu().numpy()   
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()  #维度1是
        labels_all = np.append(labels_all, labels) #叠加
        predict_all = np.append(predict_all, predic) #叠加

        if i % 10 == 0: #s多久输出一次
            train_acc = metrics.accuracy_score(labels_all, predict_all)
            kappa=metrics.cohen_kappa_score(labels_all, predict_all)
            f1=metrics.f1_score(labels_all, predict_all, average='macro')
            train_loss += loss.item()
            postfix = {'train_loss': '%.6f' % (train_loss / (i + 1)), 'kappa': '%.4f' % kappa,"train_f1":"%.4f" % f1}
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
        for i, (trains, labels) in enumerate(validate_loader, 0):  # 0是下标起始位置默认为0
   
            outputs= model(trains)
            loss = criterion(outputs, labels)
            validate_loss += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

            if i%10==0:
                acc = metrics.accuracy_score(labels_all, predict_all) #精确率
                kappa=metrics.cohen_kappa_score(labels_all, predict_all)
                f1=metrics.f1_score(labels_all, predict_all, average='macro')
                postfix = {'val_loss': '%.6f' % ( validate_loss/(i+1)), 'kappa': '%.4f' % kappa,"val_f1":"%.4f" % f1}
                validate_loader.set_postfix(log=postfix)
        
        kappa=metrics.cohen_kappa_score(labels_all, predict_all)
        precision=metrics.precision_score(labels_all, predict_all,average="macro")
        recall=metrics.recall_score(labels_all, predict_all,average="macro")
        acc = metrics.accuracy_score(labels_all, predict_all)
        f1=metrics.f1_score(labels_all, predict_all, average='macro')

        #保存结果
        try:
            save_result(config.model_save_path_final,kappa,precision,recall,f1)
        except:
            pass
        print("Kappa:%.6f  Precision:%.6f  Recall:%.6f  F1:%.6f"%(kappa,precision,recall,f1))
    validate_loader.close()
    return f1

#训练单一模型的
def single_model_main(item):
    #设置每轮训练不一样的地方
    #数据集
    config.train_path= "../dataset/emotion/emotion_train_{}.txt".format(item)
    config.dev_path="../dataset/emotion/emotion_dev_{}.txt".format(item)
    #保存路径
    config.model_save_path="./model/bert_pga_{}.pth".format(item)
    config.model_save_path_final="./model/bert_pga_final_{}.pth".format(item)
    #测试路径
    config.test_result_save_path="../dataset_solve/result/emotion_train/bert_train_{}.txt".format(item)
    config.test_path="../dataset/emotion/emotion_dev_{}.txt".format(item)

    if not os.path.exists("model"):
            os.makedirs("model")
    #加载数据集
    train_data, dev_data, _ = get_solve_dataset()
    train_iter = build_iterator(train_data)
    dev_iter = build_iterator(dev_data)
    #模型
    #model = BERTRNN()    
    model = BERT()

    #对抗训练
    if config.attack_flag is True:
        attack=PGD(model)
    else:
        attack=None
    #criterion = nn.CrossEntropyLoss() #交叉熵
    criterion=FocalLoss()#focalloss
    #判断是否要加载模型
 #   if config.load:
#        model.load_state_dict(torch.load(config.model_save_path))
    
    #BERT的参数
    bert_params = list(map(id, model.bert.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params, model.parameters())

    optimizer_grouped_parameters=[
        {"params":base_params},
        {"params":model.bert.parameters(),"lr":config.bert_lr}
    ]
    #优化器
    optimizer=AdamW(optimizer_grouped_parameters,lr=config.lr)
    scheduler=transformers.optimization.get_linear_schedule_with_warmup(optimizer,0.05*len(train_iter) * config.epoch,len(train_iter) * config.epoch)
    #训练
    val=0
    for epoch in range(config.epoch):
        #训练
        train(epoch, train_iter, config.device, model, criterion, optimizer,scheduler,attack)
        val_f1=validate(epoch,dev_iter, config.device  ,model, criterion, config.tensorboard_path)
        if val_f1>val:
  #          torch.save(model.state_dict(), config.model_save_path)
            val=val_f1
    
    #保存模型 
    torch.save(model.state_dict(), config.model_save_path_final)

if __name__ == '__main__':
    set_seed(1997)
    #对5个k折数据集进行训练
    for i in range(1,5):
        single_model_main(i)
