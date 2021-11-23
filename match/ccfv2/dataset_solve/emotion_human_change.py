#旨在人工调节结果 分类任务
#得到labellist和text_list
import pandas as pd
import os
#得到result的label_list
def get_emotion_predict_list(emotion_result_path="./result/emotion_result.txt"): #线上0.31 但是没存
    with open(emotion_result_path, 'r') as file:
        emotion_result=file.readlines()
        for i in range(0, len(emotion_result)):
            emotion_result[i] = emotion_result[i].rstrip('\n')
            emotion_result[i]=int(emotion_result[i])
    return emotion_result

#得到text_list
def get_test_text(path="../dataset/test.csv"):
    test_data = pd.read_csv(path)
    test_text_list = test_data['text'].values[:,None]
  
    text_list=[]
    for i in range(len(test_text_list)):
        text_list.append(test_text_list[i][0])
    return text_list

#按照3个类别存成3个txt
def save_emotion_result(predic_list,text_list,doc_path="./result/emotion_result_human"):
    emotion_0_path=os.path.join(doc_path,"emotion_0.txt")
    emotion_1_path=os.path.join(doc_path,"emotion_1.txt")
    emotion_2_path=os.path.join(doc_path,"emotion_2.txt")

    if os.path.exists(emotion_0_path):
        os.remove(emotion_0_path)

    if os.path.exists(emotion_1_path):
        os.remove(emotion_1_path)

    if os.path.exists(emotion_2_path):
        os.remove(emotion_2_path)

    #遍历每个文本
    for i in range(len(text_list)):
        label=int(predic_list[i])
        text=text_list[i]

        if label==0: path=emotion_0_path
        elif label==1: path=emotion_1_path
        elif label==2: path=emotion_2_path
        #追加写
        with open(path, 'a') as file:
              file.write("{}	{}".format(label,text))
              file.write("\r")
    return 
    
if __name__ == '__main__':
    #针对分类任务
    predict_result=get_emotion_predict_list()
    text_list=get_test_text()
    save_emotion_result(predict_result,text_list)
    print("end")
