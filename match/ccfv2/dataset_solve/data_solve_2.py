import pandas as pd
import random
from sklearn.model_selection import KFold 

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
train_data = pd.read_csv('../dataset/train_clean_v2.csv') #['id', 'text', 'BIO_anno', 'class',]
train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x:x.split(' '))

train_text_list = train_data['text'].values[:,None]
train_label_list = train_data['BIO_anno'].values[:,None]
train_emotion_class_list = train_data['class'].values[:,None]


#数据预处理
for i in range(len(train_text_list)):
    if i==6073:
        train_text_list[i][0]="本人近日误入广发银行的坑，以此教训!!建议大家以后慎入广发银行信用卡!!!!"
    train_text_list[i][0]=normalize(train_text_list[i][0])

def solve_data():
  
        #text
        train_x=train_text_list
        #label 
        train_y=train_label_list
        #emotion_class
        train_emotion=train_emotion_class_list

        """
        #情感训练集
        with  open('../dataset/emotion/emotion_train_test.txt', "w",encoding="utf-8") as file:
                for i in range(0,len(train_x)):
                    text_data=train_x[i][0]
                    text=text_data.replace("\t","")
                    text=text.replace(" ","")
                    file.write("{}\t{}".format(train_emotion[i][0],text))#class_label+\t+sentence
                    file.write("\r")
        """
        #NER训练集 
        with open('../dataset/NER/NER_cleanv2_all.txt', "w",encoding="utf-8") as file:
                #遍历文本
                for i in range(0,len(train_x)):
                    text_data=train_x[i]
                    #遍历每个字符
                    for j in range(len(text_data[0])):
                        try:
                            file.write("{} ".format(train_x[i][0][j])) #text
                            file.write("{}".format(train_y[i][0][j]))  #label
                            file.write('\r')
                        except:
                            print(train_x[i][0])
                            print(train_y[i][0])
                            input()
                    file.write("\r")

        return 
    



if __name__ == '__main__':
    solve_data()