import jieba
import pandas as pd
import random
from tqdm import tqdm
#emotion_classifier用的 分词后按照原有格式存成新的文件
import logging
jieba.setLogLevel(logging.INFO)
#设置种子


# 创建停用词列表
def stopwordslist(stopword_path="./stopword/cn_stopword.txt"):
     stopwords = [line.strip() for line in open(stopword_path,encoding='UTF-8').readlines()]
     return stopwords
 
# 对句子进行中文分词
def seg_depart(sentence,stopwords):
     # 对文档中的每一行进行中文分词
    print("正在分词")
    sentence_depart = jieba.cut(sentence.strip())
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += ""
    return outstr 

def fenci_solve():
    #train
    train_data = pd.read_csv('../dataset/train_clean.csv') #['id', 'text', 'BIO_anno', 'class']
    #train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x:x.split(' '))
    text_id=[]
    text_list=[]
    text_BIO=[]
    text_class=[]

    stopword_list=stopwordslist()

    for i in tqdm(range(len(train_data))):
        text_data=train_data.iloc[i]
        text_id.append(text_data["id"])

        #分词
        temp_text=text_data["text"]
        seg_str=seg_depart(temp_text,stopword_list)
        text_list.append(seg_str)

        text_BIO.append(text_data["BIO_anno"])
        text_class.append(text_data["class"])





    dataframe = pd.DataFrame({'id':text_id,'text':text_list,"BIO_anno":text_BIO,"class":text_class})
    dataframe.to_csv("../dataset/train_fenci.csv",index=None,sep=',')



if __name__ == '__main__': 
  # my_str1="哈哈哈，谁让你已经是卡圈“快升”大佬呢!!"
  # my_str2="回吗？我工行的三天了都没有回放呢,2"
   # seg_list=jieba.cut(my_str1)
   # temp=" ".join(seg_list)
   # print(temp)
  # stopword_list=stopwordslist()
  # seg_str=seg_depart(my_str2,stopword_list)
#   print(seg_str)
    fenci_solve()
