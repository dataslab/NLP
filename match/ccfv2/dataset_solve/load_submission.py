import pandas as pd
import random
from sklearn.model_selection import KFold ,StratifiedKFold
import jieba
 

#制作交叉验证集
#train
submission_data = pd.read_csv('./Submission.csv') #['id', 'BIO_anno', 'class',]

class_list=[]
for i in range(0,len(submission_data)):
        text_data=submission_data.iloc[i]
        temp=int(text_data["class"])
        class_list.append(temp)


#情感训练集
with  open('../dataset_solve/result/temp_emotion_list.txt', "w",encoding="utf-8") as file:
    for i in range(len(class_list)):
        file.write("{}".format(class_list[i]))
        file.write("\r")

print("end")
if __name__ == '__main__':
    pass