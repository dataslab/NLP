
import torch


#=================数据集==================


#train_path= "../dataset/NER/NER_train_0.txt"
train_path="../dataset/NER_all.txt"
dev_path="../dataset/NER_clean_v1/NER_dev_0.txt"
#test_path="../dataset/NER_test.txt" 没用到
#test_path="../dataset/NER_all"没用到

dataset_doc_path="../dataset"
test_ori_path="../dataset/test.csv"
#test_ori_path="../dataset/train.csv"
#=================路径====================
#bert_path="../../Bert_wwm"
bert_path="hfl/chinese-macbert-base" #macbert
#bert_path="hfl/chinese-macbert-large" #macbert

#test_result_save_path="../dataset_solve/result/NER_result.txt"
test_result_save_path="../dataset_solve/result/NER/bert_lstm_crf_4.txt"
#test_result_save_entity_path="../dataset_solve/result/NER_train_entity/train_4.txt"

model_save_path="./model/bert_large_bilistm_crf_clean_final_15epoch_0.pth"
#model_save_path_final="./model/bert_large_bilstm_crf_clean_final_20epoch_0.pth"
model_save_path_final="./model/bert_lstm_crf_final_4.pth"
tensorboard_path="./tensorboard_BERT"

#=================超参数=====================
seed=1997
bert_lr=2e-5
lr=2e-3

epoch=20
batch_size=32
#=================模型参数===================
rdrop_alpha=4
dropout_prob =0.1 
warmup_proportion=0.1
max_seq_length = 200  #单文本长度上限
num_labels=9

embedding_dim=768#1024   #bert最后是768 large是1024 small是768
hidden_dim=256  
LSTM_layers=1     
whether_lstm=True
attack_flag=False
#===========================================nvii
gpu_num=1 #gpu个数
load=False# 是否加载模型参数
device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device_list=[2,3]

#===========================================
label_list=[
    "BANK",
    "PRODUCT",
    "COMMENTS_N", #用户评论（名词）
    "COMMENTS_ADJ",#用户评论（形容词）
]

NONE = 'O'
PAD = "[PAD]"
UNK = "[UNK]"

