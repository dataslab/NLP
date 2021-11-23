import pandas as pd
import random
from sklearn.model_selection import KFold ,StratifiedKFold
import jieba
#制作形如：
def example():
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
    #以空格隔开

#设置种子
random.seed(1997)

#文本正则化 把一些生僻字什么的处理掉
def normalize(text):
    ori_text=text
    ori_text=ori_text.replace("\t","-")
    ori_text=ori_text.replace("˙","-")
    ori_text=ori_text.replace("︶","-")
    ori_text=ori_text.replace("\u3000","-")
    ori_text=ori_text.replace("\xa0","-")
 #   ori_text=ori_text.replace("‼","")
  #  ori_text=ori_text.replace(" ","-")
    return ori_text
#制作交叉验证集
#train
train_data = pd.read_csv('../dataset/train_clean.csv') #['id', 'text', 'BIO_anno', 'class',]
train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x:x.split(' '))

train_text_list = train_data['text'].values[:,None]
train_label_list = train_data['BIO_anno'].values[:,None]
train_emotion_class_list = train_data['class'].values[:,None]

#test
test_data_dic = pd.read_csv('../dataset/test.csv')

train_all_text=[]
#数据预处理
for i in range(len(train_text_list)):
    if i==6073:
        train_text_list[i][0]="本人近日误入广发银行的坑，以此教训!!建议大家以后慎入广发银行信用卡!!!!"
        train_text_list[i][0]=normalize(train_text_list[i][0])

        

def solve_data():
   # kf = KFold(n_splits=5) #5折
    #for k,(train_index,dev_index) in enumerate(kf.split(train_text_list)):
    kf=StratifiedKFold(n_splits=5,random_state=1997,shuffle=True) #换成按比例 随机打乱
    for k,(train_index,dev_index) in enumerate(kf.split(train_text_list,train_emotion_class_list)): #NER别用这个
        #text
        train_x, dev_x = train_text_list[train_index], train_text_list[dev_index] 
        #label 
        train_y, dev_y = train_label_list[train_index], train_label_list[dev_index]
        #emotion_class
        train_emotion, dev_emotion = train_emotion_class_list[train_index], train_emotion_class_list[dev_index]
  
     #   print(train_x)
    #    print(len(train_x))
  #     input()
        """
        #NER训练集 
        with open('../dataset/NER/NER_train_{}.txt'.format(k), "w",encoding="utf-8") as file:
                #遍历文本
                for i in range(0,len(train_x)):
                    text_data=train_x[i]
                    #遍历每个字符
                    for j in range(len(text_data[0])):
                        file.write("{} ".format(train_x[i][0][j])) #text
                        file.write("{}".format(train_y[i][0][j]))  #label
                        file.write('\r')
                    file.write("\r")
        #NER验证集
        with open('../dataset/NER/NER_dev_{}.txt'.format(k), "w",encoding="utf-8") as file:
                #遍历文本
                for i in range(0,len(dev_x)):
                    text_data=dev_x[i]
                    #遍历每个字符
                    for j in range(len(text_data[0])):
                        file.write("{} ".format(dev_x[i][0][j])) #text
                        file.write("{}".format(dev_y[i][0][j]))#label
                        file.write('\r')
                    file.write("\r")
        """
        #情感训练集
        with  open('../dataset/emotion/emotion_train_{}.txt'.format(k), "w",encoding="utf-8") as file:
                for i in range(0,len(train_x)):
                    text_data=train_x[i][0]
                    text=text_data.replace("\t","")
                    text=text.replace(" ","")
                    file.write("{}\t{}".format(train_emotion[i][0],text))#class_label+\t+sentence
                    file.write("\r")
        #情感验证集
        with  open('../dataset/emotion/emotion_dev_{}.txt'.format(k), "w",encoding="utf-8") as file:
                for i in range(0,len(dev_x)):
                    text_data=dev_x[i][0]
                    text=text_data.replace("\t","")
                    text=text.replace(" ","")
                    file.write("{}\t{}".format(dev_emotion[i][0],text))#class_label+\t+sentence
                    file.write("\r")
    # =============================================================================================
    

    test_data=[]

    #测试集
    for i in range(0,len(test_data_dic)):
        text_data=test_data_dic.iloc[i]
        pair_data=text_data["text"],"2" #测试标签随便设
        test_data.append(pair_data)

    """
    #情感测试集
    with  open('../dataset/emotion_test.txt', "w",encoding="utf-8") as file:
            for i in range(0,len(test_data)):
                text_data=test_data[i]
                text=text_data[0].replace("\t","")
                text=text.replace(" ","")
                file.write("{}\t{}".format(text_data[1],text))#class_label+\t+sentence
                file.write("\r") 
  
    #NER测试集
    with open('../dataset/NER_test.txt', "w",encoding="utf-8") as file:
            #遍历文本
            for i in range(0,len(test_data)):
                text_data=test_data[i]
                #遍历每个字符
                for j in range(len(text_data[0])):
                    file.write("{} ".format(text_data[0][j])) #text
                    file.write("O")#直接写O
                    file.write('\r')
                file.write("\r")
    """




    return 
    



if __name__ == '__main__':
    solve_data()