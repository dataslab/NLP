#清洗数据用的

#返回融合后的结果列表
def K_fold_data_clean():
    result_list=[]
    #读取不同折对于训练集的预测结果
    for k in range(0,5):
        emotion_result_path="./result/emotion_train/train_{}.txt".format(k)
        with open(emotion_result_path, 'r') as file:
            emotion_result=file.readlines()
            for i in range(0, len(emotion_result)):
                emotion_result[i] = emotion_result[i].rstrip('\n')
                emotion_result[i]=int(emotion_result[i])
        result_list.append(emotion_result)

    fuse_result=[]
    #遍历每个文本结果
    for i in range(len(result_list[0])):
        temp_list=[0,0,0]
        #遍历5个模型
        for j in range(len(result_list)):
            result=result_list[j][i]
            temp_list[result]+=1
        #获得最大值的索引
        result=temp_list.index(max(temp_list))
        #看看有没有221的情况
        if temp_list[0]==2 and temp_list[1]==2:
            fuse_result.append(-1)
        elif temp_list[0]==2 and temp_list[2]==2:
            fuse_result.append(-1)
        elif temp_list[1]==2 and temp_list[2]==2:
            fuse_result.append(-1)
        else:
            fuse_result.append(result)
    return fuse_result



def load_real_train_data():
    path="../dataset/emotion_train_test.txt"
    real_train_result=[]
    text=[]
    with open(path, 'r', encoding='UTF-8') as f:
        for line in f:
                line = line.strip() 
                if not line:
                    continue
                label,content = line.split("\t")   
                real_train_result.append(int(label))
                text.append(content)
    return real_train_result,text


if __name__ == '__main__':
    fuse_result_list=K_fold_data_clean()
    real_result_list,text=load_real_train_data()

    result=[]
    unusual_list=[5767, 6877, 6987, 7069, 7113, 7139, 7190, 7304, 7336]
    #遍历每一个结果 找出不一样的
    for i in range(len(fuse_result_list)):
        if fuse_result_list[i]!=real_result_list[i]:
            result.append(i)
    
    print(result)

    for i in range(len(fuse_result_list)):
        if i in unusual_list:
            print("fuse:{},real:{},text:{}".format(fuse_result_list[i],real_result_list[i],text[i]))

    for i in range(len(fuse_result_list)):
         if fuse_result_list[i]==-1:
             print(fuse_result_list[i])

    """
    fuse:0,real:2,text:中信一般不收，但是正在起诉，不过你可以帮忙
    fuse:0,real:2,text:打电话投诉，说他们催收的态度恶劣，要求银行跟你协商处理。
    fuse:0,real:2,text:那要咋办呢。蒙圈了
    fuse:0,real:2,text:周一去试试吧，人家周六周天不上班。真难
    fuse:0,real:2,text:怎么才能达到资格请指教，感觉好难
    fuse:0,real:2,text:这额度也太低了，楼主去开卡了吗
    fuse:0,real:2,text:倒霉，我申请的中行的信用卡没批
    fuse:0,real:2,text:尝试拒绝返回信用信息？这么惨？
    fuse:1,real:2,text:老哥可以啊厉害了，这个贵州银行的美团卡额度好像是独立的吧，我是听说光大的额度高，好提额，所以才想用哈，我也不知道哪些行羊毛多，额度高啊，福利好啊，老哥推荐哈...
    """
    