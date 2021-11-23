# NLP-model
- 组里一些NLP相关的代码  
- 常见的任务格式见common_format.txt

## 备注
- 每个项目记得加一个.gitignore  
- 不要上传数据集的内容  
- 过滤掉该项目的大文件部分  
- 每个项目说明一个input的格式  

## 基本环境: 
- torch=1.7  
- cudatoolkit=10.2  
- transformers=4.9.2  
- scikit-learn=0.24.2  
- tqdm


## 初学者的一些工作
NLP_begin:   
|—nlp_classification:  
| 　 　模型描述：简单的基于BERT的句子级别分类任务  
| 　 　任务类型: 分类  
| 　 　github主页: 无  
|—nlp_NER:  
| 　 　模型描述：简单的基于BERT的命名实体识别分任务  
| 　 　任务类型: 命名实体识别NER  
| 　 　github主页: 无  

## 比赛
match:  
|—ccf: 产品评论观点提取  
| 　 　模型描述：K折数据处理、模型融合(结果投票)、K折洗数据、PGA、FGM、差分学习等比赛基本策略    
| 　 　任务类型：NER+情感分类  
| 　 　比赛主页: https://www.datafountain.cn/competitions/529  


## 事件抽取
event_extraction:  
|—None

## 命名实体识别
NER:  
|—NER-LEBERT：  
| 　 　模型描述: 词典融合BERT   
| 　 　任务类型: 中文NER、中文词性标注、分词任务   
| 　 　github主页: https://github.com/liuwei1206/LEBERT  

## 机器翻译
translation:  
|—None  
