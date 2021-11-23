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

save_path="../dataset/emotion_kfold_train.txt"

def test(test_loader,device,model):
    model.eval()
    predic_list=[]
    text_list=[]
    model = model.to(device)
    test_loader = tqdm(test_loader)
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        for i, (sentence, labels) in enumerate(test_loader, 0):  # 0是下标起始位置默认为0
            outputs= model(sentence)
            predic = torch.max(outputs.data, 1)[1].cpu()
            predic_batch_list=predic.tolist()
            predic_list.extend(predic_batch_list) #添加到最终的结果中
            text_list.extend(sentence)
    test_loader.close()
    return predic_list

#改一下这个 把文本内容也加进来 要不没法进行2轮训练
def save_emotion_result(predic_list,text_list):
    #改成追加操作
    with open(save_path, "a",encoding="utf-8") as file:
        for i in range(len(predic_list)):
            file.write("{}\t{}".format(predic_list[i],text_list[i]))#
            file.write("\r") 

def load_test_text(path):
    content_list=[]
    with open(path, 'r', encoding='UTF-8') as f:
            for line in f:
                line = line.strip() 
                if not line:
                    continue
                label,content = line.split("\t")  
                content_list.append(content)
    return content_list
    

def single_model_predict(item):
    #设置每轮训练不一样的地方
  #  config.test_result_save_path="../dataset_solve/result/emotion_train/bert_train_{}.txt".format(item)
    #=================路径====================
    config.model_save_path_final="./model/bert_pga_final_{}.pth".format(item)
    config.test_path="../dataset/emotion/emotion_dev_{}.txt".format(item)

    #设置种子
    set_seed(1997)
    #加载数据集
    test_data = get_test_dataset()
    test_iter = build_iterator(test_data)
    #模型
    model=BERT()
    #model=BERTRNN()
    model.load_state_dict(torch.load(config.model_save_path_final))
    #测试
    result_list=test(test_iter,config.device,model)
    text_list=load_test_text(config.test_path)
    #print(len(result_list))
   # print(len(text_list))
    #保存结果
    save_emotion_result(result_list,text_list)


if __name__ == '__main__':
    set_seed(1997)
    #对5个k折数据集进行训练
    for i in range(0,5):
        single_model_predict(i)
