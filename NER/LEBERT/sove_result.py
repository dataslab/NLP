
def solve_result(path="./BertWordLSTMCRF_Token-Dev-378.txt"):
    label_result=[]
    with open(path, 'r',encoding="utf-8") as file:
        lines=file.readlines()
        #每一行的结果
        temp_label_str=""
        for single_line in lines :
            #说明句子结束了
            if single_line=="\n":
                label_result.append(temp_label_str)
                temp_label_str=""
                
            else:
                single_line=single_line.rstrip("\n")
                single_line_list=single_line.split("\t")
                single_label=single_line_list[2] #第二个是预测的
                #结果加入temp_label_str
                if temp_label_str!="":   
                    temp_label_str+=" "
                temp_label_str+=single_label
    return label_result

def save_result(label_list,path="./NER_result.txt"):
    with open(path, "w",encoding="utf-8") as file:
        for i in range(len(label_list)):
            file.write(label_list[i])
            file.write("\r")
    return 

if __name__ == "__main__":
    label_result=solve_result()
    save_result(label_result)
    print("end")