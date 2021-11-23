
import torch
from torch import nn 
import numpy as np
from tqdm import tqdm
import transformers
from transformers import BertConfig
from transformers.models import bert
from transformers.optimization import AdamW
from transformers import BertTokenizer
from dataset import * #数据集
import config #超参数
from model import * #模型
from utils import *
from torch.utils.data import DataLoader,SequentialSampler
import conlleval
import os

#需要写能查看30个epoch val_f1的

def train(epoch, train_loader, device, model, optimizer, scheduler,pgd,id2label):
    model.train()
    model.to(config.device)
    loss=0

    train_loader = tqdm(train_loader)
    for i,batch in enumerate(train_loader):  #(文本(数字)、mask、segment_ids(全0)、label的list(数字))
        model.train()
        batch=tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch #获取每一个
        #训练过程   
        #得到两个logits
    #    logits1= model(input_ids, input_mask) 
    #    logits2= model(input_ids, input_mask) 
        #用的Rdrop
     #   loss = Rdrop_loss(logits1,logits2,label_ids)
      #  loss.requires_grad = True
        #基本的交叉熵
      #  loss=CE_loss(logits1,label_ids)
        loss=model(input_ids, input_mask, label_ids) 
        if config.gpu_num > 1: #logit 多gpu还没有考虑logit
             loss = loss.mean() # mean() to average on multi-gpu. 
   
        loss.backward()

        #对抗训练
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
                loss_adv=model(input_ids, input_mask,label_ids)
                if config.gpu_num > 1: #logit 多gpu还没有考虑logit
                    loss = loss.mean() # mean() to average on multi-gpu. 
                loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore() # 恢复embedding参数

        optimizer.step()
        scheduler.step()
        model.zero_grad()
        loss=loss+loss.item()

        #设置显示的数据
        postfix = {"epoch": "%d"%(epoch+1), 'train_loss': '%.6f' % (loss / (i + 1))}
        train_loader.set_postfix(log=postfix)
    train_loader.close()

def validate(epoch,validate_loader,val_feature,device, model,id2label):
    model.eval()
    pred_labels=[]    #模型预测语句文本
    ori_labels = []   #真实标签文本
    ori_tokens_list=[]     #文本的token
    
    #获得原始标注ori_tokens_list
    ori_tokens_list = [f.ori_tokens for f in val_feature]
    validate_loader = tqdm(validate_loader)
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        for i,batch in enumerate(validate_loader):  #(文本(数字)、mask、segment_ids(全0)、label的list(数字)
            batch=tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch #获取每一个
            
    
            if config.gpu_num==1:
            #    logits= model(input_ids, input_mask)   #//得到logits
                logits= model.predict(input_ids, input_mask)   #//得到logits #crf的是偶
            else:
      #          logits= model(input_ids, input_mask)   #//得到logits
                logits= model.module.predict(input_ids, input_mask)
          
            #将logits转换成标签(id2label)
            preds=logits #crf要这个
      #      preds = np.argmax(logits.cpu().numpy(), axis=2).tolist() #crf不需要这个
            for l in preds:
                pred_labels.append([id2label[idx] for idx in l])  
                        
            for l in label_ids:
                ori_labels.append([id2label[idx.item()] for idx in l])
          

              
            #计算各个评价指标 
            eval_list = [] 
            for ori_tokens, oril, prel in zip(ori_tokens_list, ori_labels, pred_labels):
                    for ot, ol, pl in zip(ori_tokens, oril, prel):
                        if ot in ["[CLS]", "[SEP]"]:
                            continue
                        eval_list.append(f"{ot} {ol} {pl}\n")
                    eval_list.append("\n")

            # eval the model 
            counts = conlleval.evaluate(eval_list)
            #conlleval.report(counts)
            overall, by_type = conlleval.metrics(counts)
            #为了实时显示结果
            postfix = {"epoch": "%d"%(epoch+1), 'val_prec': '%.4f' % overall.prec,"val_f1":"%.4f" % overall.fscore}
            validate_loader.set_postfix(log=postfix)
        
        # eval the model 
        counts = conlleval.evaluate(eval_list)
        #conlleval.report(counts)
        overall, by_type = conlleval.metrics(counts)
        conlleval.report(counts)

        #保存结果
        save_result(config.model_save_path_final,overall.prec,overall.rec,overall.fscore)

    validate_loader.close()
    return overall.fscore


def single_model_train(item):
    config.train_path= "../dataset/NER_clean_v1/NER_train_{}.txt".format(item)
    config.dev_path="../dataset/NER_clean_v1/NER_dev_{}.txt".format(item)
    #保存路径
    config.model_save_path="./model/bert_lstm_crf_15epoch_{}.pth".format(item)
    config.model_save_path_final="./model/bert_lstm_crf_final_{}.pth".format(item)
    #测试路径
    config.test_result_save_path="../dataset_solve/result/NER_train/bert_crf_train_{}.txt".format(item)
    config.test_path="../dataset/NER/NER_dev_{}.txt".format(item)

   
    label_list,label2id,id2label=build_vocab(config.label_list) #获得所有标签的列表、labeltoid id2label
    
    tokenizer=None
    #构建模型+bert参数
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
   # model=BertSoftmaxForNer(num_labels=len(label_list))
    model=BERT_BiLSTM_CRF(num_labels=len(label_list))
    #对抗训练
    if config.attack_flag is True:
        attack=PGD(model)
    else:
        attack=None
    
    if config.load:
        model.load_state_dict(torch.load(config.model_save_path))
    
    config.gpu_num=1
    
    if config.gpu_num > 1:
           model = torch.nn.DataParallel(model,device_ids=config.device_list)
    
    #加载数据集 
    train_features,train_data = get_SolveDataset(mode="train",label2id=label2id,tokenizer=tokenizer)
    train_dataloader = DataLoader(train_data,sampler=SequentialSampler(train_data), batch_size=config.batch_size)
    
   
    val_features,val_data = get_SolveDataset(mode="val",label2id=label2id,tokenizer=tokenizer)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data),batch_size=config.batch_size)
 
    #差分学习率
    bert_params = list(map(id, model.bert.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params, model.parameters())

    optimizer_grouped_parameters=[
        {"params":base_params},
        {"params":model.bert.parameters(),"lr":config.bert_lr}
    ]

    #优化器
    optimizer=AdamW(optimizer_grouped_parameters,lr=config.lr)
    #sheduler
    scheduler=transformers.optimization.get_linear_schedule_with_warmup(optimizer,0,len(train_data) * config.epoch)

    #训练
    val=0
    for epoch in range(config.epoch):
        #训练
        train(epoch, train_dataloader, config.device, model, optimizer,scheduler,attack,id2label)
        val_f1=validate(epoch, val_dataloader,val_features, config.device, model,id2label)
        #if val_f1>val:
        if epoch==14:
            model_to_save = model.module if hasattr(model, 'module') else model 
            torch.save(model_to_save.state_dict(), config.model_save_path) #这个
    #保存模型 
    torch.save(model.state_dict(), config.model_save_path_final)

if __name__ == '__main__':

    if not os.path.exists("model"):
        os.makedirs("model")
        
    set_seed(config.seed)   #设置种子
    for item in range(0,5): #5折
        single_model_train(item)
   


