import config
import os
import torch
from tqdm import tqdm
import logging
import pickle
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

#数据集的格式 
"""
彭 B-name 
小 I-name 
军 I-name 
认 O 
为 O 
， O 
国 O 
内 O 
银 O 
行 O 
现 O 
在 O 

我 O 
们 O 
阿 O 
森 O 
纳 O 
是 O
不 O
可 O
战 O
胜 O
滴 O
"""

class _InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text 
        self.label = label

class _InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, label_id, ori_tokens):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.ori_tokens = ori_tokens

#NER任务最后根据数据得到的label集合
def get_label(path=config.dataset_doc_path):
    labels = set() 
    #如果找到了之前存好的类的pkl
    if os.path.exists(os.path.join(path, "label_list.pkl")):
            logger.info(f"loading labels info from {path}")
            with open(os.path.join(path, "label_list.pkl"), "rb") as f:
                labels = pickle.load(f)
    else:
        logger.info(f"loading labels info from train file and dump in {path}")
        with open(os.path.join(path, "train_input.txt"), encoding="utf-8") as f:
                    for line in f.readlines():
                        tokens = line.strip().split(" ")

                        if len(tokens) == 2:
                            labels.add(tokens[1])

        if len(labels) > 0:
            with open(os.path.join(path, "label_list.pkl"), "wb") as f:
                        pickle.dump(labels, f)
        else:
            logger.info("loading error and return the default labels B,I,O")
            labels = {"O", "B", "I"}
            
    return labels 

#加载文本
def _load_dataset(path):
    """Reads a BIO data."""
    with open(path, "r", encoding="utf-8") as f:
        lines = []  #["label_seq","text_seq"]
        words = []
        labels = []
        for line in tqdm(f.readlines()):   
            contends = line.strip()
            tokens = line.strip().split(" ")

            if len(tokens) == 2: #读出数据
                    words.append(tokens[0])
                    labels.append(tokens[1])
            else:
                if len(contends) == 0 and len(words) > 0:
                        label = []
                        word = []
                        for l, w in zip(labels, words):  #一句话结束了 将其加入lines
                            if len(l) > 0 and len(w) > 0:
                                label.append(l)
                                word.append(w)
                        lines.append([' '.join(label), ' '.join(word)])
                        words = []
                        labels = []
    return lines

#加载数据并处理格式
def _get_examples(path):
        examples = []  
        lines =_load_dataset(path)
        for i, line in enumerate(lines):
            guid = str(i)
            text = line[1]
            label = line[0]
            examples.append(_InputExample(guid=guid, text=text, label=label))
        return examples
 
#将数据转换成bert需要的格式 
def _convert_examples_to_features(examples,label2id, tokenizer,max_seq_length=config.max_seq_length): #labellist是get_label的结果
   # label_map = {label : i for i, label in enumerate(label2id)} #构建label与对应的index的集合
    label_map=label2id
    features = []

    for (ex_index, example) in tqdm(enumerate(examples), desc="convert examples"): #example(guide、text、label)

        textlist = example.text.split(" ")
        labellist = example.label.split(" ")
        assert len(textlist) == len(labellist)
        tokens = []
        labels = []
        ori_tokens = []

        for i, word in enumerate(textlist):
            # 防止wordPiece情况出现，不过貌似不会
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            ori_tokens.append(word)

            # 单个字符不会出现wordPiece
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    if label_1 == "O":
                        labels.append("O")
                    else:
                        labels.append("I")
            
        if len(tokens) >= max_seq_length - 1:  #采用直接截断的方式 舍弃超过该长度的内容
            tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
            labels = labels[0:(max_seq_length - 2)]
            ori_tokens = ori_tokens[0:(max_seq_length - 2)]

        ori_tokens = ["[CLS]"] + ori_tokens + ["[SEP]"]  #添加bert标记
        
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_ids.append(label_map["O"])

        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])

        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_ids.append(label_map["O"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)   #将token转换成索引
        
        input_mask = [1] * len(input_ids)

        assert len(ori_tokens) == len(ntokens), f"{len(ori_tokens)}, {len(ntokens)}, {ori_tokens}"

        while len(input_ids) < max_seq_length: #补齐剩余部分
            input_ids.append(0)
            input_mask.append(0)  
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        features.append(
                _InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              ori_tokens=ori_tokens))

    return features


def get_SolveDataset(mode,label2id,tokenizer):
    if mode == "train":   path = config.train_path
    elif mode == "val":  path = config.dev_path
    elif mode == "test":  path = config.test_path
   
    examples = _get_examples(path) #获得数据
    features = _convert_examples_to_features(examples,label2id,tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(config.device)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(config.device)
 #  all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(config.device)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(config.device)

    data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)  #(文本(数字)、mask、label的list(数字))

    return features,data


if __name__ == '__main__':
    #load_dataset(config.train_path)
    labellist=get_label(config.dataset_doc_path)
