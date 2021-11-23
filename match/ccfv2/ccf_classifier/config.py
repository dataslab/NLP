
import torch


#=================数据集==================
#train_path= "../dataset/emotion/emotion_train_1.txt"
#train_path= "../dataset/emotion_train_test.txt"
train_path= "../dataset/emotion_kfold_train.txt"
dev_path="../dataset/emotion/emotion_dev_0.txt"

test_path="../dataset/emotion_test.txt"
#test_path="../dataset/emotion_train_test.txt"
#=================路径====================

bert_path="hfl/chinese-macbert-base" #macbert
model_save_path="./model/bert_pga_all_kfold_10epoch.pth"
model_save_path_final="./model/bert_pga_all_kfold_15epoch.pth"
tensorboard_path="./tensorboard_BERT"
test_result_save_path="../dataset_solve/result/emotion_result.txt"
#test_result_save_path="../dataset_solve/result/emotion_train/train_clean_0.txt"
#=================超参数==================
seed=1997
bert_lr=2e-5
lr=2e-3
epoch=15
batch_size=32
dropout_prob=0.1 
#=================模型参数===================
pad_size = 200  #单文本长度上限
embedding_dim=768 #bert最后是768
hidden_dim=128
LSTM_layers=1     
num_classes=3      #分类任务的分类个数
#===========================================
load=False# 是否加载模型参数
attack_flag=True #是否进行梯度对抗
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")