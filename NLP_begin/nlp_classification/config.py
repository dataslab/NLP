
import torch


#=================数据集==================
train_path= "./dataset/train.txt"
test_path="./dataset/test.txt"
dev_path="./dataset/validation.txt"
#=================路径====================
bert_path="../Bert_wwm"
model_save_path="./bert_LSTM_model.pth"
tensorboard_path="./tensorboard_BERT"
#=================超参数==================
seed=1997
lr=5e-5
epoch=3
batch_size=16
dropout_prob=0.2 
#=================模型参数===================
pad_size = 96  #单文本长度上限
embedding_dim=768 #bert最后是768
hidden_dim=512
LSTM_layers=2     
num_classes=2      #分类任务的分类个数
#===========================================
load=False# 是否加载模型参数
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")