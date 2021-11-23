import torch 
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

#保存测试集准确率
def save_result(model_name,kappa,precision,recall,f1):
    if not os.path.exists("model_result"):
        os.makedirs("model_result")
    #name=./model/bert_base_final_drop.pth
    name=model_name.split("/")[-1]
    formal_name=name.split(".")[0]
    path="./model_result/{}.txt".format(formal_name)
    with open(path, "a") as f: #追加在末尾
        f.write("{:.4f}, {:.4f}, {:.4f},  {:.4f}".format(kappa, precision, recall ,f1))
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


#多文本分类的focalloss
class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''
    def __init__(self, gamma=2, weight=None, balance_param=1,ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index=ignore_index
        self.balance_param = balance_param


    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        focal_loss = F.nll_loss(logpt, target, self.weight,ignore_index=self.ignore_index)

        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss



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



#=====================对抗训练==========================
import torch

# grad_backup = {}


# def save_grad(tensorName):
#     def backward_hook(grad: torch.Tensor):
#         grad_backup[tensorName] = grad

#     return backward_hook


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
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]




class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



if __name__ == '__main__':
    # # 示例过程 PGD
    # pgd = PGD(model)
    # K = 3 # 小步走的步数
    # for batch_input, batch_label in data:
    #     # 正常训练
    #     loss = model(batch_input, batch_label)
    #     loss.backward() # 反向传播，得到正常的grad
    #     pgd.backup_grad()

    #     # 对抗训练
    #     for t in range(K):
    #         pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
    #         if t != K-1:
    #             model.zero_grad()
    #         else:
    #             pgd.restore_grad()
    #         loss_adv = model(batch_input, batch_label)
    #         loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    #     pgd.restore() # 恢复embedding参数

    #     # 梯度下降，更新参数
    #     optimizer.step()
    #     model.zero_grad()



    # # 示例过程 FGM
    # fgm = FGM(model)
    # for batch_input, batch_label in data:
    #     # 正常训练
    #     loss = model(batch_input, batch_label)
    #     loss.backward()  # 反向传播，得到正常的grad
    #     # 对抗训练
    #     fgm.attack()  # 在embedding上添加对抗扰动
    #     loss_adv = model(batch_input, batch_label)
    #     loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
    #     fgm.restore()  # 恢复embedding参数
    #     # 梯度下降，更新参数
    #     optimizer.step()
    #     model.zero_grad()
    pass