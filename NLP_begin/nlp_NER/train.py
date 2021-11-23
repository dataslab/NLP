
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

def train(epoch, train_loader, device, model, optimizer, scheduler,id2label):
    model.train()
    loss=0
    train_loader = tqdm(train_loader)
    
    for i,batch in enumerate(train_loader):  #(文本(数字)、mask、segment_ids(全0)、label的list(数字))
        model.train()
      
        batch=tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch #获取每一个
        #训练过程
        optimizer.zero_grad()
        #loss,logits= model(input_ids, input_mask, label_ids)   #//得到训练过程中的损失值、logits 不在这得到logits了 时间成本高
        loss= model(input_ids, input_mask, label_ids) 
        if config.gpu_num > 1: #logit 多gpu还没有考虑logit
             loss = loss.mean() # mean() to average on multi-gpu. 
             
        loss.backward()
  
        optimizer.step()
        scheduler.step()
        loss+=loss.item()

        #设置显示的数据
       # postfix = {"epoch": "%d"%(epoch+1), 'train_loss': '%.6f' % (loss / (i + 1)), 'train_prec': '%.4f' % overall.prec,"train_f1":"%.4f" % overall.fscore}
       # train_loader.set_postfix(log=postfix)
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
                
            logits= model.predict(input_ids, input_mask)   #//得到logits
         
            #将logits转换成标签(id2label)
            for l in logits:
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
    validate_loader.close()
    return overall.fscore

if __name__ == '__main__':

    if not os.path.exists("model2"):
        os.makedirs("model2")
        
    set_seed(config.seed)   #设置种子
    label_list=get_label() #获得所有标签的列表
    config.num_labels=len(label_list) #从而设置模型最后需要分出几类 模型构造的时候要用
    #获得label2id 和id2label
    if os.path.exists(os.path.join(config.dataset_doc_path, "label2id.pkl")):
            with open(os.path.join(config.dataset_doc_path, "label2id.pkl"), "rb") as f:
                label2id = pickle.load(f)
    else:
        label2id = {l:i for i,l in enumerate(label_list)}  
        with open(os.path.join(config.dataset_doc_path, "label2id.pkl"), "wb") as f:
            pickle.dump(label2id, f)      
    id2label = {value:key for key,value in label2id.items()} 
    
    print(label2id)
    print(id2label)
    tokenizer=None
    #构建模型+bert参数
    if config.load:
        bert_config=BertConfig.from_pretrained(config.bert_model_save_path,num_labels=len(label_list))
        model=BERT_BiLSTM_CRF.from_pretrained(config.bert_model_save_path,config=bert_config,num_labels=len(label_list))
        tokenizer=BertTokenizer.from_pretrained(config.bert_model_save_path) #这个得在加载数据前面。。。。
    else:
        bert_config=BertConfig.from_pretrained(config.bert_path,num_labels=len(label_list))
        model=BERT_BiLSTM_CRF.from_pretrained(config.bert_path,config=bert_config,num_labels=len(label_list))
        tokenizer = BertTokenizer.from_pretrained(config.bert_path)

    model.to(config.device)
    if config.gpu_num > 1:   model = torch.nn.DataParallel(model)
    
    #加载数据集 
    train_features,train_data = get_SolveDataset(mode="train",label2id=label2id,tokenizer=tokenizer)
    train_dataloader = DataLoader(train_data,sampler=SequentialSampler(train_data), batch_size=config.batch_size)
    val_features,val_data = get_SolveDataset(mode="val",label2id=label2id,tokenizer=tokenizer)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data),batch_size=config.batch_size)
 
    #BERT训练时需要改变的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    #优化器
    optimizer=AdamW(optimizer_grouped_parameters,lr=config.lr)
    #sheduler
    scheduler=transformers.optimization.get_linear_schedule_with_warmup(optimizer,0.05*len(train_data) * config.epoch,len(train_data) * config.epoch)

    #训练
    val=0
    for epoch in range(config.epoch):
        #训练
        train(epoch, train_dataloader, config.device, model, optimizer,scheduler,id2label)
        val_f1=validate(epoch, val_dataloader,val_features, config.device, model,id2label)
        if val_f1>val:
            model_to_save = model.module if hasattr(model, 'module') else model 
            torch.save(model_to_save.state_dict(), config.model_save_path) #这个不管用 不知道为啥 存进去读不出来
            model_to_save.save_pretrained(config.bert_model_save_path)   
            tokenizer.save_pretrained(config.bert_model_save_path)
            val=val_f1

    #保存模型 
    torch.save(model.state_dict(), config.model_save_path)


