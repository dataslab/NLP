import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np
import os
from config import NONE, PAD
import config

#保存测试集准确率
def save_result(model_name,precision,recall,f1):
    if not os.path.exists("model_result"):
        os.makedirs("model_result")
    #name=./model/bert_base_final_drop.pth
    name=model_name.split("/")[-1]
    formal_name=name.split(".")[0]
    path="./model_result/{}.txt".format(formal_name)
    with open(path, "a") as f: #追加在末尾
        f.write("{:.4f}, {:.4f},  {:.4f}".format(precision, recall ,f1))
        f.write('\n') #换行
        f.close()



#设置种子
def set_seed(seed):
    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) #禁止hash随机化
    #if torch.cuda.is_available():
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  #并行gpu
    torch.backends.cudnn.deterministic = True  #cpu/gpu结果一致
    torch.backends.cudnn.benchmark = False  
    torch.backends.cudnn.enabled = False        #非确定性算法 会加快 但是无法复现

def build_vocab(labels, BIO_tagging=True):
    all_labels = ["O"]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label

#Rdrop
"""
def loss_fnc(self, y_pred, y_true, alpha=4):
        #配合R-Drop的交叉熵损失

        loss1 =nn.CrossEntropyLoss(y_pred, y_true)
        loss2 =nn.KLDivLoss(reduction="none")(torch.log_softmax(y_pred[::2], dim=1), y_pred[1::2].softmax(dim=-1)) + \
               nn.KLDivLoss(reduction="none")(torch.log_softmax(y_pred[1::2], dim=1), y_pred[::2].softmax(dim=-1))

        return loss1 + torch.mean(loss2) / 4 * alpha
"""


def Rdrop_loss(logits1, logits2, label, alpha=4):

      #  [16,200,9]
 
        """
        配合R-Drop的交叉熵损失
        """
        #交叉熵损失
        #ce_loss = 0.5 * (F.cross_entropy(logits1, label,ignore_index=0) + F.cross_entropy(logits2, label,ignore_index=0))
        ce_loss = 0.5 * (F.cross_entropy(logits1.view(-1,config.num_labels), label.view(-1)) + \
                         F.cross_entropy(logits2.view(-1,config.num_labels), label.view(-1)))
        #KL散度
        p_loss = F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction='none')
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()
        kl_loss = (p_loss + q_loss) / 2

        return ce_loss + kl_loss * alpha


def Rdrop_loss_crf(_logits1, _logits2, label, alpha=4):
        """
        配合R-Drop的交叉熵损失
        """
        temp_logits1=np.array(_logits1)
        print(_logits1)
        print(temp_logits1)
        print(temp_logits1.size())
        logits1=torch.from_numpy(temp_logits1)
        temp_logits2=np.array(_logits2)
        logits2=torch.from_numpy(temp_logits2)
        #交叉熵损失
        #ce_loss = 0.5 * (F.cross_entropy(logits1, label,ignore_index=0) + F.cross_entropy(logits2, label,ignore_index=0))
        ce_loss = 0.5 * (F.cross_entropy(logits1.view(-1,config.num_labels), label.view(-1)) + \
                         F.cross_entropy(logits2.view(-1,config.num_labels), label.view(-1)))
        #KL散度
        p_loss = F.kl_div(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
        q_loss = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1), reduction='none')
        p_loss = p_loss.mean()
        q_loss = q_loss.mean()
        kl_loss = (p_loss + q_loss) / 2

        return ce_loss + kl_loss * alpha

#普通的交叉熵损失函数
def CE_loss(logits1,label):
    ce_loss = F.cross_entropy(logits1.view(-1,config.num_labels), label.view(-1))
    return ce_loss


class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self,
               epsilon=1.,
               alpha=0.3,
               emb_name='emb.',
               is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        # 此处也可以直接用一个成员变量储存 grad，而不用 register_hook 存储在全局变量中
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    param.grad = self.grad_backup[name]


def extraction_entity(text_list,label_list):
    dic={}
    #遍历所有的结果
    for i in range(len(text_list)):
        single_label_list=label_list[i]
        single_text_list=text_list[i]
    
        #遍历所有的label
        for j in range(len(single_label_list)):
            label=single_label_list[j]     
            try:
                char=single_text_list[j]
            #跳过有问题的
            except:
                print(len(single_label_list))
                print(len(single_text_list))
                print(single_label_list)
                print(single_text_list)
                continue
          #      input()
            #如果是O就跳过
            if label=="O":
                continue
            else:
                bio_index=label.split("-")[0]
                entity_name=label.split("-")[1]
                if bio_index=="B":
                    temp_word=""+char
                    #向后遍历
                    k=j+1
                    #小于语句的长度  
                    while k<len(single_label_list):
                        #后边的位置的label
                        temp_label=single_label_list[k]
                        if temp_label=="O":
                            break
                        temp_char=single_text_list[k]
                        temp_bio_index=temp_label.split("-")[0]
                        temp_entity_name=temp_label.split("-")[1]
                        if temp_bio_index=="I" and temp_entity_name==entity_name:
                            temp_word+=temp_char
                            k+=1
                        else:
                            break

                    #将temp_word插入dic中
                    if entity_name not in dic:
                        dic[entity_name]=[]
                    dic[entity_name].append(temp_word)       

    #去重
    for key in dic.keys():
        temp_list=dic[key]
        temp_set=set(temp_list)
        temp_list=list(temp_set)
        dic[key]=temp_list


    return dic
import json
def save_entity(dic,save_path):
    s=json.dumps(dic)
    with open(save_path, 'w') as file:
        file.writelines(s)
    return 

