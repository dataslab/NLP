
import pandas as pd
# -*- coding:utf-8 -*-
def load_data():
    train_data = pd.read_csv('../dataset/train_clean.csv',encoding="utf-8") #['id', 'text', 'BIO_anno', 'class', 'bank_topic']
    train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x:x.split(' '))
    test_data = pd.read_csv('../dataset/test.csv') #id,text

    return train_data,test_data

#测试文本长度
def test_length_text(data):
    max_length=0
    length_list=[0 for _ in range(10000)]
    max_list=[]
    for i in range(0,len(data)):
        text_data=data.iloc[i]
        text_length=len(text_data["text"])
        length_list[text_length]+=1
        if text_length>max_length:
            max_length=text_length
        if text_length>200:
            max_list.append(text_length)
    print(max_list)
        
    return length_list,max_length
# 测试类别比例
#训练集：#[242, 138, 9620]

def test_emoition(data):
    my_list=[0,0,0]
    for i in range(0,len(data)):
        text_data=data.iloc[i]
        temp=int(text_data["class"])
        my_list[temp]+=1
    return my_list
        
#训练集   
#样本超过200的文本长度 [258,210,278,352]
#超过128长度的有[199, 143, 161, 258, 210, 278, 352, 200, 151, 147, 150, 153, 151, 157, 155, 151, 154]

#测试集超过128的文本长度有
"""
[219, 197, 141, 606, 241, 161, 150, 139, 142, 173, 150, 1042, 382, 2101, 621, 251, 140, 
187, 1826, 902, 2182, 133, 153, 292, 735, 319, 357, 232, 1521, 4942, 535, 3923, 580, 1403, 199, 
182, 204, 1650, 294, 543, 8869, 173, 226, 231, 4278, 349, 2169, 166, 590, 129, 770, 280, 2760, 476,
135, 137, 400, 274, 815, 168, 415, 161, 210, 354, 723, 294, 784, 503, 1225, 152, 136, 491, 138, 374, 
6893, 964, 1691, 174, 196, 166, 384, 328, 152, 142, 189, 1086, 336, 144, 171, 174, 164, 129, 255, 153,
628, 986, 240, 479, 2902, 1606, 317, 278, 320, 234, 5338, 193, 1797, 1741, 214, 361, 521, 154, 170, 938, 
198, 280, 154, 209, 651, 450, 169, 242, 1885, 1168, 135, 1060, 227, 399, 1817, 147, 1185, 162, 158, 2333,
351, 177, 159, 144]
"""


if __name__ == '__main__':
    train_dic,test_dic=load_data()
    emotion_list=test_emoition(train_dic)
    print(emotion_list)
    #length_list,max_length=test_length_text(test_dic)
    #print(length_list)
    #print(max_length) #最长的文本长度为352
    
