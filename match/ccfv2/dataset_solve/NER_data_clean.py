import pandas as pd
from tqdm import tqdm
import json
import re 
label_list=[['O', 'O', 'O', 'O', 'B-COMMENTS_N', 'I-COMMENTS_N', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-COMMENTS_N', 'I-COMMENTS_N', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-COMMENTS_ADJ', 'I-COMMENTS_ADJ', 'O', 'B-BANK', 'I-BANK', 'O', 'O', 'O', 'O', 'O', 'B-COMMENTS_N', 'I-COMMENTS_N', 'B-COMMENTS_ADJ', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']]
text_list=[['共', '享', '一', '个', '额', '度', '，', '没', '啥', '必', '要', '，', '四', '个', '卡', '不', '要', '年', '费', '吗', '？', '你', '这', '种', '人', '头', '，', '银', '行', '最', '喜', '欢', '，', '广', '发', '是', '出', '了', '名', '的', '风', '控', '严', '，', '套', '现', '就', '给', '你', '封', '.', '.', '.']]

#返回两个list相交的元素
def return_repeat_list_item(list1,list2):
    result_list=[]
    set1=set(list1)
    set2=set(list2)

    set3=set1&set2
    result_list=list(set3)
    return result_list


#比如这个没必要 应该是漏标的
#text_list 和label_list都是二维数组

def solve_train_csv():
    train_data = pd.read_csv('../dataset/train_clean.csv') #['id', 'text', 'BIO_anno', 'class',]
    train_data['BIO_anno'] = train_data['BIO_anno'].apply(lambda x:x.split(' '))
    train_text_list = train_data['text'].values[:,None]
    train_label_list = train_data['BIO_anno'].values[:,None]
  
    label_list=[]
    text_list=[]
    for i in range(len(train_label_list)):
        label_list.append(train_label_list[i][0])
        text_list.append(train_text_list[i][0])
  #  print(label_list)

    return text_list,label_list
  
def extraction_entity(text_list,label_list):
    dic={}
    #遍历所有的结果
    for i in range(len(text_list)):
        single_label_list=label_list[i]
        single_text_list=text_list[i]
    
        #遍历所有的label
        for j in range(len(single_label_list)):
            label=single_label_list[j]     
            try:
                char=single_text_list[j]
            #跳过有问题的
            except:
                print(len(single_label_list))
                print(len(single_text_list))
                print(single_label_list)
                print(single_text_list)
                continue
          #      input()
            #如果是O就跳过
            if label=="O":
                continue
            else:
                bio_index=label.split("-")[0]
                entity_name=label.split("-")[1]
                if bio_index=="B":
                    temp_word=""+char
                    #向后遍历
                    k=j+1
                    #小于语句的长度  
                    while k<len(single_label_list):
                        #后边的位置的label
                        temp_label=single_label_list[k]
                        if temp_label=="O":
                            break
                        temp_char=single_text_list[k]
                        temp_bio_index=temp_label.split("-")[0]
                        temp_entity_name=temp_label.split("-")[1]
                        if temp_bio_index=="I" and temp_entity_name==entity_name:
                            temp_word+=temp_char
                            k+=1
                        else:
                            break

                    #将temp_word插入dic中
                    if entity_name not in dic:
                        dic[entity_name]=[]
                    dic[entity_name].append(temp_word)                  
    return dic

def save_entity(dic,save_path):
    s=json.dumps(dic)
    with open(save_path, 'w') as file:
        file.writelines(s)
    return 

def load_entity(save_path):
    with open(save_path, 'r') as file:
        lines=file.readline()
        dic=json.loads(lines)
    return dic


#分期不可以降负债的：工商，建设，农行，光大，招行，华夏，浦发，宁波

#找出同时出现在K折预测结果中的实体
def K_fold_data_clean(dic_list):
    result_dic={}
    temp_dic=dic_list[0]
    for key in temp_dic.keys():
        result_dic[key]=[]
    
    #遍历每个实体,筛选K折中均出现的实体
    for key in result_dic.keys():
        single_set=set(dic_list[0][key])
        #遍历每个dic  #找出交集
        for i in range(1,len(dic_list)):
            temp_set=set(dic_list[i][key])
            single_set=single_set&temp_set
        result_list=list(single_set)
        result_dic[key]=result_list
   # print(result_dic)
    return result_dic

#找出训练集中没有在k_fold结果中出现的实体 漏标
def search_miss_data(k_fold_dic,train_dic):
    result_dic={}
    for key in k_fold_dic.keys():
        result_dic[key]=[]
    
    for key in result_dic.keys():
        for i in range(len(k_fold_dic[key])):
            if k_fold_dic[key][i] not in train_dic[key]:
                result_dic[key].append(k_fold_dic[key][i])

    print(result_dic)
    return

#找出训练集标注但是k-fold没预测的
def search_wrong_data(k_fold_dic,train_dic):
    result_dic={}
    for key in k_fold_dic.keys():
        result_dic[key]=[]
    
    for key in result_dic.keys():
        for i in range(len(train_dic[key])):
            if train_dic[key][i] not in k_fold_dic[key]:
                result_dic[key].append(train_dic[key][i])

    print(result_dic)
    return

#找出dic中不同类的实体中重复的元素()
def search_repeat_entity(entity_dic):
    temp_list=[]
    result_list=[]
    #遍历字典
    for key in entity_dic.keys():
        #遍历实体
        for entity in entity_dic[key]:
            #如果没发现实体 先把它放在temp_list
            if entity not in temp_list:
                temp_list.append(entity)
            else:
                result_list.append(entity)
  #  print(result_list)
    temp_set=set(result_list)
    result_list=list(temp_set)
    return result_list

#找出实体中个数更多的那个种类
def search_more_type(entity_list,entity_dic,text_list,label_list):
    repeat_dic={} #记录实体的两个种类
    num_dic={}  #记录实体的数量
    result_list=[]
    #返回重复实体数量多的那个key
    #遍历重复的实体
    #找出重复的两个键

    #entity_list去重
    temp_set=set(entity_list)
    entity_list=list(temp_set)

    for entity in entity_list:
        #遍历键
        key_1="temp"
        key_2="temp"
        flag=False
        for key in entity_dic.keys():
            #如果找到了
            if entity in entity_dic[key]:
            #第一个实体类别
                if flag==False:
                    flag=True
                    key_1=key
                else:
                    key_2=key
        #找到这两个键
        repeat_dic[entity]=(key_1,key_2)
    
    for key in repeat_dic.keys():
         num_dic[key]=(0,0)

 #   print(repeat_dic)
    #遍历所有训练文本        
    for i in tqdm(range(len(text_list))):
        text=text_list[i]
        single_label_list=label_list[i]
        #遍历所有实体
        for j in range(0,len(entity_list)):   
                entity=entity_list[j]
                key_1,key_2=repeat_dic[entity] 
                num1,num2=0,0
                #找到该实体在文本中的下标list
                entity_index_list=[k.start() for k in re.finditer(entity, text)]
                #找到了实体
                if len(entity_index_list)>0:
                    for entity_index in entity_index_list:
                        if single_label_list[entity_index]=="B-{}".format(key_1):
                            num1+=1
                        elif single_label_list[entity_index]=="B-{}".format(key_2):
                            num2+=1
                num1=num_dic[entity][0]+num1
                num2=num_dic[entity][1]+num2

                num_dic[entity]=(num1,num2)
  #  print(num_dic)
    #给出高的那个种类
    equation_list=[]
    
    for entity in entity_list: 
        num1,num2 = num_dic[entity]
        if num1==num2:
            equation_list.append(entity)
        elif num1>num2:
            result_list.append((entity,repeat_dic[entity][0]))
        else:
            result_list.append((entity,repeat_dic[entity][1]))
    
  #  print(equation_list)
    #print(result_list)
    return result_list

#文本中出现比较少的实体 返回一个小于出现阈值的实体列表 并返回过滤好的dic
def dic_threshold(text_list,label_list,dic,num=10,persent=0.2): #耗时比较长
    #这里的label_list是分割好的
    entity_list=[] #包含全部实体的列表
    entity_num_list=[]
    entity_all_list=[]

    result_list=[]
    persent_list=[]

    result_dic={}
    for key in dic.keys():
        result_dic[key]=[]
        entity_list.extend(dic[key])
    #去重
    temp_set=set(entity_list)
    entity_list=list(temp_set)

    #初始化实体num
    for i in range(len(entity_list)):
        entity_num_list.append(0)
        entity_all_list.append(0)

    #工行
    #工行卡这种 短的那个实体出现的次数就都算了

    #遍历所有文本
    for i in tqdm(range(len(text_list))):
        text=text_list[i]
        single_label_list=label_list[i]
        #遍历实体列表
        for j in range(0,len(entity_list)):
            entity=entity_list[j]
            #找到该实体在文本中的下标list
            entity_index_list=[k.start() for k in re.finditer(entity, text)]
            if len(entity_index_list)>0:
                #entity_num加上长度
                for entity_index in entity_index_list:
                    #如果label_list中不是O才计数    
                    if single_label_list[entity_index]!="O":
                        entity_num_list[j]+=1 
                    entity_all_list[j]+=1

    #遍历记录实体name的        
    for i in range(len(entity_num_list)):
        if entity_num_list[i]<num:
            result_list.append(entity_list[i])
    
    #返回百分比list
    for i in range(len(entity_num_list)):
        single_percent=(entity_num_list[i]/entity_all_list[i])
        if single_percent<persent:
            persent_list.append(entity_list[i])
    print(persent_list)
    return result_list

#替换有实体标注但是数据集中漏标的 #如果重复的话则把按照最高的  再人工定一个list 比如支付-支付宝-支付宝口碑 支付宝无法全覆盖
def replace_single_tag_entity(label_list,text_list,entity_dic,long_entity_list=None,force_replace_list=None): 
    #K是阈值 文本中
    #entity_dic={"PRODUCT":["网贷","可以"]}
    #text_list=["网贷都可以，想办都可以申请网贷"]
    #label_list=["O", "O","O", O O O O O O O O O O B-PRODUCT I-PRODUCT"]
    
    #long_entity_list #不需要覆盖的文本的标签为O
   # force_replace_list #强制覆盖 
 
    result_label_list=[]
    #遍历每条data
    for data_index in tqdm(range(0,len(text_list))):
        text=text_list[data_index]
       # label=label_list[data_index]
        single_label_list=label_list[data_index]
   #     single_label_list=label.split(" ")
        #遍历每个实体
        for key in entity_dic.keys():
            #对每个实体都扫一遍
            for entity in entity_dic[key]:
                #计算出实体的长度
                entity_length=len(entity)
                #找到该实体在文本中的下标list
                entity_index_list=[i.start() for i in re.finditer(entity, text)]
                #遍历这些下标
                for index in entity_index_list: 
                    #如果是长实体的话 "工行卡" 完全覆盖
                    if long_entity_list is not None and  entity  in long_entity_list:
                        single_label_list[index]="B-"+key    
                        for i in range(1,entity_length):
                            single_label_list[index+i]="I-"+key    
                    else:
                        #如果是短实体的话，只有是O的才覆盖
                        if single_label_list[index]=="O":
                            single_label_list[index]="B-"+key
                        for i in range(1,entity_length):
                            if single_label_list[index+i]=="O":
                                single_label_list[index+i]="I-"+key
        
        if force_replace_list is not None:
            #强制替换的实体 比如支付-支付宝-支付宝口碑 可能被覆盖成支付宝了 需要把支付宝口碑进行覆盖
            #遍历每个实体
            for key in entity_dic.keys():
                #对每个实体都扫一遍
                for entity in entity_dic[key]:
                    #如果不是在强制替换里
                    if entity not in force_replace_list:
                        continue
                    #计算出实体的长度
                    entity_length=len(entity)
                    #找到该实体在文本中的下标list
                    entity_index_list=[i.start() for i in re.finditer(entity, text)]
                    #遍历这些下标
                    for index in entity_index_list:
                        single_label_list[index]="B-"+key    
                        for i in range(1,entity_length):
                            single_label_list[index+i]="I-"+key    
        result_label_list.append(single_label_list)
    return result_label_list

#"工行卡"-->"工行"这种
def search_repeat_long_entity(dic):
    entity_list=[]
    long_entity_list=[]  #长实体列表
    small_entity_list=[]   #短实体列表
    for key in dic.keys():
        entity_list.extend(dic[key])
    

    #遍历每个待寻找的实体
    for search_entity in entity_list:
        #遍历字典中每个键
        for key in dic.keys():
            #遍历每个实体
            for entity in dic[key]:
                #不是自己
                if search_entity==entity:
                    continue
                #匹配字符串 看entity中是否包含search_entity
                if(entity.find(search_entity))!=-1: #包含
                    long_entity_list.append(entity)
                    small_entity_list.append(search_entity)

    
    #entity_list去重
    temp_set=set(long_entity_list)
    long_entity_list=list(temp_set)
    temp_set=set(small_entity_list)
    small_entity_list=list(temp_set)

    return long_entity_list,small_entity_list
    
#人工修订
def artificial_dic():
    dic={}
    dic["降额"]="PRODUCT"
    dic["工行卡"]="PRODUCT"
    dic["支付"]="COMMENTS_N"
    dic["有水"]="COMMENTS_ADJ"
    dic["利润"]="COMMENTS_N"
    dic["翻车"]="COMMENTS_ADJ"
    #========================
    dic["融易借"]="PRODUCT"
    dic["频率"]="COMMENTS_N"
    dic["返现"]="COMMENTS_N"
    return dic

#有些实体人工消除，不进行操作
def artificial_delete_list():
    result_list=[]
    result_list.append("消")
    result_list.append("标注")
    result_list.append("分期")
    result_list.append("结清")
    result_list.append("黑户")
    result_list.append("开卡")
    result_list.append("北京")
    result_list.append("中国")
    result_list.append("上海")
    result_list.append("中原")
    result_list.append("东亚")
    temp_list=['宁波', '广州',"试试"]
    result_list.extend(temp_list)
    
    #从标注小于10次中筛出来的部分实体
    temp_list=["试试","通过","算了","最后","量大","不赚钱","尽快","支持","幅度","随缘","弊端","预计",
    "忽略","坚持","不介意","运气","一样","不一样"]
    result_list.extend(temp_list)
    temp_list=["不影响","不可以", "协商","千万","操作","算了","等待","秒批","不给","一直","别想","一般", "影响",
    "这么","等等","不信", "取消", "希望", "幅度", "通过","明白","肯定","只是","能不能","调整","不一样","一样",
    "哈哈","省","不过","福利","运气","等待","没啥","随便","主动",
    "没有","最后","赶紧","真实","明确", "不要","感谢","减免","无法","大概率","清楚","不清楚"]
    result_list.extend(temp_list)
    
    #去重
    temp_set=set(result_list)
    result_list=list(result_list)

    return result_list
#寻找多层嵌套实体
def search_more_nesting_entity(long_entity_list):
    mid_list=[]
    last_list=[] #相当于顶层的 需要在替换的时候强制覆盖
    #如果在long_entity_list中还能匹配到自己,那么它就是 支付-支付宝-支付宝口碑中的支付宝
    #遍历实体
    for entity in long_entity_list:
        #该实体遍历long_entity_list:
        for search_entity in long_entity_list:    
            if(search_entity.find(entity))!=-1: 
                if search_entity!=entity:
                    mid_list.append(entity)
                    last_list.append(search_entity)
    
    #去重
    temp_set=set(mid_list)
    mid_list=list(temp_set)
    return last_list 

#寻找长度短的实体 不进行操作 返回不处理的
def search_small_length_entity(dic):
    result_list=[]
    result_dic={}
    for key in dic.keys():
        result_dic[key]=[]

    for key in dic.keys():
        for entity in dic[key]:
                if len(entity)==1:
                    result_list.append(entity)
                    result_dic[key].append(entity)
    #print(result_list)
 #   print(result_dic)
    #把有歧义的都不考虑了
    """
   {'BANK': ['兴', '设', '汇', '邮', '交', '浦', '中', '建', '农', '信', '光', '平', '工', '民', '华', '招'], 
   'COMMENTS_N': [], 
   'COMMENTS_ADJ': ['升', '跌', '牛', '死', '怒',
   '棒', '懒', '傻', '急', '差', '长', '爱', '害', '亏', '仇', '烦', '猛', '虚', '好', '乱',
   '怂', '停', '屁', '高',  '减', '怕', '累', '假', '喜','悬',
   '愁', '懵', '靓', '黑', '满', '慌', '涨', '蠢', '臭', '敢', '蒙',
   '瞎', '慢', '赞', '省', '废', '批', '香', '疯', '难', '穷', '烂', '贱', '惨', '爽',  '晕',
   '严', '负', '快', '稳', '哭', '贵', '胜', '低', '拒', '狠', '坑', '降'],
    'PRODUCT': ['盾', '卡']}
    """
    
    #处理这些
    human_list=[]
    human_list.extend(['升', '跌', '牛', '死', '怒',
   '棒', '懒', '傻', '急', '差', '长', '爱', '害', '亏', '仇', '烦', '猛', '虚', '好', '乱',
   '怂', '停', '屁', '高',  '减', '怕', '累', '假', '喜','悬',
   '愁', '懵', '靓', '黑', '满', '慌', '涨', '蠢', '臭', '敢', '蒙',
   '瞎', '慢', '赞', '省', '废', '批', '香', '疯', '难', '穷', '烂', '贱', '惨', '爽',  '晕',
   '严', '负', '稳', '哭', '贵', '胜', '低', '拒', '狠', '坑', '降'])
    human_list.extend( ['盾', '卡'])


    #求出在result_list不在human_list里的
    set1=set(result_list)
    set2=set(human_list)
    set3=set1-set2
    result_list=list(set3)
    #返回这些 让dic删除掉这些key
    ['平', '太', '真', '小', '邮', '行', '又', '套', '晚', '最', '旧', '光', '不', '没', '交', '信', 
    '设', '水', '近', '工', '少', '华', '贫', '浦', '汇', '民', '农', '提', '多', '空', '下', '中', '才', 
    '示', '深', '建', '富', '消', '拖', '骗', '赚', '补', '大', '怼', '兴', '爆', '掉', '招', '错', '久']

    return result_list

#处理threthold的 以字典形式返回
def solve_threshold_dic(dic,threshold_list):
    result_dic={}
    for key in dic.keys():
            result_dic[key]=[]
    for key in dic.keys():
        for entity in dic[key]:
            if entity in threshold_list:
                result_dic[key].append(entity)
    print(result_dic)
#处理得到最终的dic
def solve_final_dic(train_dic):
    result_dic=train_dic
    #找出重复的实体
    repeat_entity_list=search_repeat_entity(train_dic)

    #找出重复实体中更多的那个种类 以元组的形式返回
    change_pair_list=search_more_type(repeat_entity_list,train_dic,text_list,label_list)

    #先删除train_dic中重复的实体
    for repeat_key in repeat_entity_list:
         #遍历字典里面的实体值
        for key in result_dic.keys():
            #重复的键被发现的话就删除
            if repeat_key in result_dic[key]:
                result_dic[key].remove(repeat_key)
    
    #把实体加入到指定字典中
    for entity,key in change_pair_list:
        result_dic[key].append(entity)
    
    #加入人工修订的dic
    human_dic=artificial_dic()

    #如果有 先把原来的结果删除了
    for key in result_dic.keys():
        for entity in human_dic.keys():
            if entity in result_dic[key]:
                result_dic[key].remove(entity)

    for key in human_dic.keys():
        result_dic[human_dic[key]].append(key)


    #删除模糊的人工修订的list
    small_entity_list=search_small_length_entity(result_dic)
    
    for repeat_key in small_entity_list:
         #遍历字典里面的实体值
        for key in result_dic.keys():
            #重复的键被发现的话就删除
            if repeat_key in result_dic[key]:
                result_dic[key].remove(repeat_key)


    #加入人工需要删除的list
    human_list=artificial_delete_list()
    
    for key in result_dic.keys():    
        for entity in human_list:
            if entity in result_dic[key]:
                result_dic[key].remove(entity)
   # print(result_dic)
    return result_dic


#存成比赛要用的csv形式
def save_csv(text_list,label_list):
    #text_list可以直接用，label_list需要每条拼接一下
    final_label_list=[]
    id_list=[]
    class_list=[]
 #   print(len(label_list))
    for single_label_list in label_list:
        single_list=""
        for i in range(len(single_label_list)-1):
            single_list+=single_label_list[i]
            single_list+=" "
        single_list+=single_label_list[-1]
        final_label_list.append(single_list)

    #['id', 'text', 'BIO_anno', 'class']
    train_data = pd.read_csv('../dataset/train_clean.csv',encoding="utf-8") 
    for i in range(0,len(train_data)):
        text_data=train_data.iloc[i]
        single_class=int(text_data["class"])
        single_id=int(text_data["id"])

        id_list.append(single_id)
        class_list.append(single_class)
    #存成csv文件
    #id,text,BIO_anno,class
    dataframe = pd.DataFrame({"id":id_list,"text":text_list,"BIO_anno":final_label_list,"class":class_list})
    dataframe.to_csv("../dataset/train_clean_v2.csv", index=None,sep=',')
    print("end")



if __name__ == '__main__':
    
    text_list,label_list=solve_train_csv()
    #抽取实体
    train_dic=extraction_entity(text_list,label_list)

    #遍历字典里面的实体值
    for key in train_dic.keys():
        temp_list=train_dic[key]
        temp_set=set(temp_list)
        temp_list=list(temp_set)
        train_dic[key]=temp_list

    """  
    #同时出现在K折预测结果中的实体
    dic_list=[]
    for i in range(0,1):
        path="./result/NER_train_entity/train_{}.txt".format(i)
        result_dic=load_entity(path)
        dic_list.append(result_dic)
    k_fold_dic=K_fold_data_clean(dic_list)
    """
    #search_miss_data(k_fold_dic,train_dic)
    #search_wrong_data(k_fold_dic,train_dic)
    
    #找到小于阈值的实体
    #threshold_list=dic_threshold(text_list,label_list,train_dic)
  #  print(load_entity("./threshold_list.txt"))
   # save_entity(threshold_list,"./threshold_list_2.txt")
    threshold_list=load_entity("./threshold_list_5.txt")

    #处理一些特殊的情况
    train_dic=solve_final_dic(train_dic)   
    print(train_dic)
    #找到"工商卡"->"工商这种实体"
    long_entity_list,small_entity_list=search_repeat_long_entity(train_dic)
    #需要强制替换的实体
    force_replace_list=search_more_nesting_entity(long_entity_list)


    #solve_threshold_dic(train_dic,threshold_list)
    
    #强制替换
    #label_result=replace_single_tag_entity(label_list,text_list,train_dic,long_entity_list=long_entity_list,force_replace_list=force_replace_list)
    #save_entity(label_result,"./label_result.txt")
    label_result=load_entity("./label_result.txt")
    save_csv(text_list,label_result)
    
