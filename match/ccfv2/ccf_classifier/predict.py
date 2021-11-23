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


def test(test_loader,device,model):
    model.eval()
    predic_list=[]
    model = model.to(device)
    test_loader = tqdm(test_loader)
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        for i, (sentence, labels) in enumerate(test_loader, 0):  # 0是下标起始位置默认为0
            outputs= model(sentence)
            predic = torch.max(outputs.data, 1)[1].cpu()
            predic_batch_list=predic.tolist()
            predic_list.extend(predic_batch_list) #添加到最终的结果中
    test_loader.close()
    return predic_list


def save_emotion_result(predic_list):
    if os.path.exists(config.test_result_save_path):
        os.remove(config.test_result_save_path)
    with open(config.test_result_save_path, "w",encoding="utf-8") as file:
        for i in range(len(predic_list)):
            file.write("{}".format(predic_list[i]))#
            file.write("\r\n") 

if __name__ == '__main__':
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
    #保存结果
    save_emotion_result(result_list)

    print("end")