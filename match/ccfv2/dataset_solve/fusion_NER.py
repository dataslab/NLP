import os 
#得到融合的结果
def get_K_fold_data(path="./result/NER"):
    #单条文本str"没分割结果" 每个item是一个label_result
    result_list=[]
    #读取不同折对于训练集的预测结果
    for k in range(0,1):
        NER_result_path=os.path.join(path,"WCBertlstmCRF_{}.txt".format(k))
        with open(NER_result_path, 'r') as file:
            single_NER_label_result=file.readlines()
            for i in range(len(single_NER_label_result)):
                single_NER_label_result[i] = single_NER_label_result[i].rstrip('\n')
        result_list.append(single_NER_label_result)
    return result_list


#处理融合结果
def solve_fuse_result_list(fuse_result_list,entity_list):
    result_label=[]
    #遍历每条文本
    for i in range(len(fuse_result_list[0])):
        single_result_label=[]
        #每条文本
        single_temp_list=[]
        #遍历5个结果
        for single_label_result_list in fuse_result_list:
            single_label_list=single_label_result_list[i]
            #以" "划分
            label_list=single_label_list.split(" ")
            single_temp_list.append(label_list)
 
 
        #投票！取最多的那个
        #遍历label_list的每个字符
        for j in range(len(single_temp_list[0])):
            vote_num_list=[0 for _ in range(len(entity_list))]
            #遍历每种结果
            for result_index in range(len(single_temp_list)):
                label=single_temp_list[result_index][j]
                index=entity_list.index(label)
                vote_num_list[index]+=1  #增加个数
            maxlabel_index = vote_num_list.index(max(vote_num_list))
            maxlabel=entity_list[maxlabel_index]
            single_result_label.append(maxlabel)

        #把label_list变成str
        single_text=""
        for j in range(len(single_result_label)-1):
            single_text+=single_result_label[j]
            single_text+=" "
        single_text+=single_result_label[-1]
      
        result_label.append(single_text)
    return result_label

        
def build_vocab(labels, BIO_tagging=True):
    all_labels = ["O"]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}
    return all_labels, label2idx, idx2label

def save_NER_result(label_list,path="./result/NER_result.txt"):
    with open(path, "w",encoding="utf-8") as file:
        for i in range(len(label_list)):
            text_result=label_list[i] 
            file.write("{}".format(text_result))
            file.write("\r") 

if __name__ == '__main__':
    entity_list=[
    "BANK",
    "PRODUCT",
    "COMMENTS_N", #用户评论（名词）
    "COMMENTS_ADJ",#用户评论（形容词）
    ]
    #获得NER的label_list
    entity_list, _ , _=build_vocab(entity_list)
    
    fuse_result_list=get_K_fold_data()
    #K折文本融合
    label_result_list=solve_fuse_result_list(fuse_result_list,entity_list)
    save_NER_result(label_result_list)
    print("end")
    
