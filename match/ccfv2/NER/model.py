
from torch import nn 
import config
import torch.nn.functional as F
from transformers import BertModel,BertPreTrainedModel
from torchcrf import CRF
from transformers import BertConfig
import transformers
#NER任务的BERT+BiLSTM+CRF

#BERT+SoftMax
class BertSoftmaxForNer(nn.Module):
    def __init__(self, num_labels):
        super(BertSoftmaxForNer, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout_prob) 
        self.classifier = nn.Linear(config.embedding_dim,num_labels)
 
    def forward(self, input_ids, input_mask):
        encoder_out, _  = self.bert(input_ids = input_ids,attention_mask=input_mask,return_dict=False)
        encoder_out = self.dropout(encoder_out)
        logits = self.classifier(encoder_out)
        outputs = logits 
        return outputs  



class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, num_labels=config.num_labels, 
        hidden_dim=config.hidden_dim,num_layers=config.LSTM_layers,whether_lstm=config.whether_lstm):
        super(BERT_BiLSTM_CRF, self).__init__()
        out_dim = config.embedding_dim  #bert的维度
        self.flag = whether_lstm
        self.bert = transformers.BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout_prob) #dropout

        # 如果为False，则不要BiLSTM层  #lstm
        if self.flag:
            self.lstm = nn.LSTM(config.embedding_dim, hidden_dim, num_layers, bidirectional=True, batch_first=True)
            out_dim = hidden_dim*2
        
        self.hidden2tag = nn.Linear(out_dim, num_labels)   #classifier
        self.crf = CRF(num_labels, batch_first=True)       #crf
       
    
    #得到bert出来的logits(emussions) 传入crf
    def _tag_outputs(self, input_ids, input_mask=None):
        encoder_out, _ = self.bert(input_ids, attention_mask=input_mask,return_dict=False)  
        if self.flag:
            sequence_output, _ = self.lstm(encoder_out)
            sequence_output = self.dropout(sequence_output)
        else:
             sequence_output = self.dropout(encoder_out)
        emissions = self.hidden2tag(sequence_output)
        return emissions

    #前向传播
    def forward(self, input_ids, input_mask,label_ids):# 
        emissions = self._tag_outputs(input_ids, input_mask)    #用这个logits如何
        loss = -1*self.crf(emissions, label_ids, mask=input_mask.byte())
        return loss

    #crf预测结果
    def predict(self, input_ids, input_mask=None):
        emissions = self._tag_outputs(input_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())




if __name__ == '__main__':
   pass
    