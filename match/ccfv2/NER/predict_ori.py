
import torch
from torch import nn 
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from transformers import BertConfig,BertTokenizer
from transformers.utils.dummy_pt_objects import load_tf_weights_in_convbert
from dataset import * #数据集
import config #超参数
from model import * #模型
from utils import *
from torch.utils.data import DataLoader,SequentialSampler
import conlleval
import pandas as pd


def test(test_loader, test_feature, device, model,id2label,tokenizer):
    model.eval()
    pred_labels=[]    #模型预测语句文本
    result_label=[]   #最终返回的文本
    all_ori_tokens = [f.ori_tokens for f in test_feature]    #ori_tokens是加入了"[CLS]", "[SEP]"

    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        for i,batch in enumerate(test_loader):  #(文本(数字)、mask、、label的list(数字)
            batch=tuple(t.to(device) for t in batch)
            input_ids, input_mask, label_ids = batch #获取每一个
          #调试
          #  text_ids=input_ids[2].tolist()
          #  text=tokenizer.convert_ids_to_tokens(text_ids)
            if config.gpu_num==1:
                logits= model.predict(input_ids, input_mask)   #//得到logits
            else:
                logits= model(input_ids, input_mask)   #//得到logits

            preds=logits
            #preds = np.argmax(logits.cpu().numpy(), axis=2).tolist()
            #对batch中的预测结果进行操作 得到的句子中包括"[CLS]", "[SEP]"
            for l in preds:
                pred_label = []
                for idx in l:
                    pred_label.append(id2label[idx]) 
                pred_labels.append(pred_label)


    result_text=[] #原始文本
    #处理"[CLS]","[SEP]"        
    for ori_tokens,pred_label in zip(all_ori_tokens,pred_labels): #sentence
        temp_label=[]
        temp_text=[]
        for ot,pl in zip( ori_tokens,pred_label):#charactor 
            if ot in ["[CLS]", "[SEP]"]:
                continue
            else:
                temp_label.append(pl)
                temp_text.append(ot)
            
        result_label.append(temp_label)
        result_text.append(temp_text)

        print(result_label)
        print(result_text)
        input()
        break
    #输出所有实体
    entity_dic=extraction_entity(result_text,result_label)
    #保存文本
    save_entity(entity_dic,config.test_result_save_entity_path)
    return  result_label

#保存结果
def save_NER_result(pred_labels):
    if os.path.exists(config.test_result_save_path):
        os.remove(config.test_result_save_path)
    with open(config.test_result_save_path, "w",encoding="utf-8") as file:
        for i in range(len(pred_labels)):
            text_result=pred_labels[i] 
            text=""
            for j in range(0,len(text_result)-1): 
                text=text+text_result[j]+" "
            text+=text_result[-1]
            file.write("{}".format(text))#
            file.write("\r\n") 

def get_test_str(test_path=config.test_ori_path):
    test_data_dic = pd.read_csv(test_path,encoding="utf-8")
    test_data=[]
    #测试集
    for i in range(0,len(test_data_dic)):
        text_data=test_data_dic.iloc[i]
        test_data.append(text_data["text"])
    return test_data

if __name__ == '__main__':

    set_seed(config.seed)   #设置种子
    label_list,label2id,id2label=build_vocab(config.label_list) #获得所有标签的列表、labeltoid id2label


    #模型初始化
    bert_config=BertConfig.from_pretrained(config.bert_path,num_labels=len(label_list))
    tokenizer=BertTokenizer.from_pretrained(config.bert_path)
  #  model=BertSoftmaxForNer.from_pretrained(config.bert_path,config=bert_config,num_labels=len(label_list))
    model=BERT_BiLSTM_CRF(num_labels=len(label_list))

    model.to(config.device)
    
    #用单卡预测
    config.gpu_num=1 
    
    if config.gpu_num > 1:
        model = torch.nn.DataParallel(model,device_ids=[1,2,3])
    
    #加载参数
    model.load_state_dict(torch.load(config.model_save_path_final))

    #加载数据集 
    test_list=get_test_str() #
 
    result_list=[]
    
    input()
    #训练
    test_len=0
    for i in tqdm(range(0,len(test_list))):
        test_features,test_data = get_SingleData(test_list[i],label2id=label2id,tokenizer=tokenizer,max_size=config.max_seq_length,)
        test_dataloader = DataLoader(test_data,sampler=SequentialSampler(test_data), batch_size=1)
        single_text_list=test(test_dataloader, test_features, config.device, model,id2label,tokenizer)
        #将single_text_list拼接
        single_text=[]
        for j in range(0,len(single_text_list)):
            single_text.extend(single_text_list[j])
            #print("{} +len:{}".format(single_text_list[j],len(single_text_list[j])))
            test_len+=len(single_text_list[j])

        #label的长度为1082
    
        result_list.append(single_text)
 
    save_NER_result(result_list)
