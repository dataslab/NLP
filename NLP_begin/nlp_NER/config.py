
import torch


#=================数据集==================
train_path= "./dataset/train_input.txt"
test_path="./dataset/dev_input.txt"
dev_path="./dataset/dev_input.txt"
dataset_doc_path="./dataset"
#=================路径====================
bert_path="../Bert_wwm"
model_save_path="./bert_LSTM_model.pth"
bert_model_save_path="./model3"
tensorboard_path="./tensorboard_BERT"
#=================超参数==================
seed=1997
lr=3e-5
epoch=3
batch_size=16
dropout_prob=0.2 
#=================模型参数===================
max_seq_length = 300  #单文本长度上限
embedding_dim=768 #bert最后是768
hidden_dim=128  #512
LSTM_layers=1     
num_labels=25 #标签个数 #这个是要根据那个label_list的长度算出来的  #改过了这里不需要也没事
whether_lstm=True
#===========================================
gpu_num=1 #gpu个数
load=False# 是否加载模型参数
device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu") #这里改GPU编号