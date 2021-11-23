import config
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
tokenizer = BertTokenizer.from_pretrained(config.bert_path)

#数据集的格式 label+"\t"+sentence
#1	死囚爱刽子手女贼爱衙役我们爱你们难道还有别的选择没想到胡军除了蓝宇还有东宫西宫我去阿兰这样真他nia恶心爱个P分明只是欲

#加载数据集  
def load_dataset(path,pad_size=64): #参数:数据集路径、文本长度限制 
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                line = line.strip() 
                if not line:
                    continue
                label,content = line.split("\t")   
                token = tokenizer.tokenize(content) #bert的分词
                token = [CLS] + token #添加bert的标志
                seq_len = len(token) 
                mask = []   #掩码，注意力机制
                token_ids = tokenizer.convert_tokens_to_ids(token)      #word2index #

                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token)) #前面是1 剩余部分补0
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        #采用直接截断的方式抛弃多余文本
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size

                contents.append((token_ids, int(label), seq_len, mask)) #每条数据为文本wordtoindex后的list、标签、长度、mask
    np.random.shuffle(contents)
    return contents #返回文本wordtoindex后的list、标签、长度、mask

#加载处理数据集、将数据和标签转换
def get_solve_dataset():
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train,dev,test

class DatasetIterater(object):
    def __init__(self, dataset_content, batch_size, device):  #content、批大小、和是否是gpu训练。  
        #content为文本wordtoindex后的list、标签、长度、mask
        self.batch_size = batch_size
        self.batches = dataset_content
        self.n_batches = len(dataset_content) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(dataset_content) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        #=============================================================
        return (x, seq_len, mask), y    #文本、长度、mask、标签
        #=============================================================
    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
            
    def __getitem__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches
    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset_content):
    iter = DatasetIterater(dataset_content, config.batch_size, config.device)
    return iter


if __name__ == '__main__':

   train,dev,test=get_solve_dataset()
   print(len(train))