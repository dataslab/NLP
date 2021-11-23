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


def test(test_loader,device,model,criterion,tensorboard_path):

    model.eval()
    test_loss = 0.0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    model = model.to(device)
    test_loader = tqdm(test_loader)
    with torch.no_grad():  # 进行评测的时候网络不更新梯度
        for i, (tests, labels) in enumerate(test_loader, 0):  # 0是下标起始位置默认为0
            outputs= model(tests)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

            if i%10==0:
                acc = metrics.accuracy_score(labels_all, predict_all) #精确率
                f1=metrics.f1_score(labels_all, predict_all, average='macro')
                postfix = {'test_loss': '%.6f' % ( test_loss/(i+1)), 'test_acc': '%.4f' % acc,"test_f1":"%.4f" % f1}
                test_loader.set_postfix(log=postfix)

        precision=metrics.precision_score(labels_all, predict_all,average="macro")
        recall=metrics.recall_score(labels_all, predict_all,average="macro")
        acc = metrics.accuracy_score(labels_all, predict_all)
        f1=metrics.f1_score(labels_all, predict_all, average='macro')

    print("\n")
    print("Accuracy:%.6f  Precision:%.6f  Recall:%.6f  F1:%.6f"%(acc,precision,recall,f1))
    print("test_end")
    test_loader.close()

if __name__ == '__main__':
     #设置种子
    set_seed(1997)

    #加载数据集
    _, _, test_data = get_solve_dataset()
    test_iter = build_iterator(test_data)
    #模型
    model=BERTRNN()
    criterion = nn.CrossEntropyLoss() #交叉熵
    model.load_state_dict(torch.load(config.model_save_path))
    #测试
    test(test_iter,config.device,model,criterion,config.tensorboard_path)
