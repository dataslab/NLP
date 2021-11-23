
import torch
from torch import nn 
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from transformers import BertConfig
from transformers import BertTokenizer
from transformers.utils.dummy_pt_objects import load_tf_weights_in_convbert
from dataset import * #数据集
import config #超参数
from model import * #模型
from utils import *
from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
import conlleval


def test(test_loader, test_feature, device, model,id2label):
    model.eval()
    pred_labels=[]    #模型预测语句文本
    ori_labels = []   #真实标签文本
    ori_tokens_list=[]     #文本的token
    
    #获得原始标注ori_tokens_list
    ori_tokens_list = [f.ori_tokens for f in test_feature]
    test_loader = tqdm(test_loader)
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        for i,batch in enumerate(test_loader):  #(文本(数字)、mask、、label的list(数字)
            batch=tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch #获取每一个
                
            logits= model.predict(input_ids, input_mask)   #//得到logits
         
            #转换成标签
            for l in logits:
                pred_labels.append([id2label[idx] for idx in l])  
                        
            for l in label_ids:
                ori_labels.append([id2label[idx.item()] for idx in l])
          
              
            #计算各个评价指标
            test_list = []
            for ori_tokens, oril, prel in zip(ori_tokens_list, ori_labels, pred_labels):
                    for ot, ol, pl in zip(ori_tokens, oril, prel):
                        if ot in ["[CLS]", "[SEP]"]:
                            continue
                        test_list.append(f"{ot} {ol} {pl}\n")
                    test_list.append("\n")

            # eval the model 
            counts = conlleval.evaluate(test_list)
            overall, by_type = conlleval.metrics(counts)
            postfix = {'test_prec': '%.4f' % overall.prec,"test_f1":"%.4f" % overall.fscore}
            test_loader.set_postfix(log=postfix)
        
        # eval the model 
        counts = conlleval.evaluate(test_list)
        overall, by_type = conlleval.metrics(counts)
        conlleval.report(counts)
    test_loader.close()
    return 

if __name__ == '__main__':

    set_seed(config.seed)   #设置种子
    label_list=get_label() #获得所有标签的列表 #这个获得的集合之前是tm随机的
   
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
    
   
    label_map = {label : i for i, label in enumerate(label_list)} 
   
    print(label_map)
    print(label2id)
    print(id2label)


    #从保存模型的地方读取
    bert_config=BertConfig.from_pretrained(config.bert_model_save_path,num_labels=len(label_list))
    tokenizer=BertTokenizer.from_pretrained(config.bert_model_save_path)
    model=BERT_BiLSTM_CRF.from_pretrained(config.bert_model_save_path,config=bert_config,num_labels=len(label_list))

    model.to(config.device)
    if config.gpu_num > 1:   model = torch.nn.DataParallel(model)

    #加载数据集 
    test_features,test_data = get_SolveDataset(mode="test",label2id=label2id,tokenizer=tokenizer)
    test_dataloader = DataLoader(test_data,sampler=SequentialSampler(test_data),batch_size=config.batch_size)
  
    model.load_state_dict(torch.load(config.model_save_path)) #这个不好用 g了
    #训练
    test(test_dataloader, test_features,config.device, model,id2label)

    #id2label 别tm设置成全局变量！ 
    

