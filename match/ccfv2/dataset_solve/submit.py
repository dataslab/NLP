import pandas as pd
import datetime
import os
import shutil
#path
ner_result_path="./result/NER_result.txt"   #目前线上0.40
emotion_result_path="./result/emotion_result.txt"  #线上0.31 但是没存
#emotion_result_path="./result/human_change_emotion_result.txt"  #人工修订的
ner_result=[]
emotion_result=[]
#NER_result
with open(ner_result_path, 'r') as file:
    temp_ner_result=file.readlines()
    for i in range(0, len(temp_ner_result)):
        temp_ner_result[i] = temp_ner_result[i].rstrip('\n')
        ner_result.append(temp_ner_result[i])
#都标记成"O" 测试
    """
    for i in range(0,len(ner_result)):
  
        temp_list=ner_result[i].split(" ")
        single_result=""
        for j in range(len(temp_list)-1):
            single_result+="O"
            single_result+=" "
        single_result+="O"
        ner_result[i]=single_result
    """
#统计情感各有多少
emotion_num=[0,0,0]

#emotion_result
with open(emotion_result_path, 'r') as file:
    emotion_result=file.readlines()
    for i in range(0, len(emotion_result)):
        emotion_result[i] = emotion_result[i].rstrip('\n')
        emotion_result[i]=int(emotion_result[i])
        emotion_num[emotion_result[i]]+=1
   #     emotion_result[i]=2  #都弄成2测试一下

print(emotion_num)
#id_list
id_list=[]
for i in range(0,len(ner_result)):
    id_list.append(i)

#生成submit
dataframe = pd.DataFrame({"id":id_list,"BIO_anno":ner_result,"class":emotion_result})
dataframe.to_csv("Submission.csv", index=None,sep=',')

#生成submit的备份文件
now_time = datetime.datetime.now().strftime("%m-%d %H:%M:%S")
doc_path="./history/{}".format(now_time)
os.makedirs(doc_path)
dataframe.to_csv(os.path.join(doc_path,"Submission.csv"), index=None,sep=',')
shutil.copy(ner_result_path, doc_path)
shutil.copy(emotion_result_path, doc_path)



#动态调整
def dictionary():
    pass
    #融易借