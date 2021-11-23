import torch
from torch import nn 
import config
import torch.nn.functional as F
import transformers


class BERT(nn.Module):  
    def __init__(self):
        super(BERT, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.bert_path)  #bert预训练模型
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(config.dropout_prob)
        self.fc = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, x):  #参数x表达的内容为：(context(wordtoindex后的list), seq_len, mask)  (x, seq_len, mask)
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, pooled = self.bert(context, attention_mask=mask,return_dict=False)
        out = self.dropout(pooled)
        out = self.fc(pooled)
        return out


#pooling层只取了最后一个时间序列，也可以取所有的时间序列

class BERTRNN(nn.Module):
    def __init__(self,embedding_dim=config.embedding_dim, 
    hidden_dim=config.hidden_dim,num_layers=config.LSTM_layers,num_classes=config.num_classes):
        super(BERTRNN, self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim = hidden_dim      #维度
        self.num_layers=num_layers        #单元个数
        self.num_classes=num_classes

        self.bert =  transformers.BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, self.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout_prob)
        self.dropout = nn.Dropout(config.dropout_prob)
        self.fc_rnn = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, x):   #(x, seq_len, mask)
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask,return_dict=False)  
        out, _ = self.lstm(encoder_out)  #(output),(h_n,c_n)      tensor的格式(batch , sequence_length , feature)  因为batch_first为true 否则为(seq,batch,feature)

        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc_rnn(out[:, -1, :])  #取句子最后时刻的 hidden state   
        return out

    """
    lstm:  #(output),(h_n,c_n) output是最后一层lstm的每个词向量对应隐藏层的输出
    output=(seq_length,batch_size,num_directions*hidden_size)
    hn,cn是所有层最后一个隐藏元和记忆元的输出，和层数、隐层大小有关
    """





if __name__ == '__main__':
    model = BERT()
    model = BERTRNN()
    