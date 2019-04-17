import torch
import torch.autograd as autograd
import torch.nn as nn 
import torch.optim as optim
from torchcrf import CRF

torch.manual_seed(1)


START_TAG = "<START>"
STOP_TAG = "<END>"

class EM_BiLSTM_CRF(nn.Module):

    def __init__(self, words_size, divs_size , chs_size, embedding_dim, hidden_dim,feature_dim = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.words_size = words_size
        self.tagset_size1 = divs_size
        self.tagset_size2 = chs_size

        self.word_embeds = nn.Embedding(words_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,num_layers=1, bidirectional=True , batch_first=True) 

        self.linear = nn.Linear(hidden_dim,feature_dim)

        self.hidden2tag1 = nn.Linear(feature_dim, self.tagset_size1)
        self.hidden2tag2 = nn.Linear(feature_dim , self.tagset_size2)

        self.div_crf = CRF(self.tagset_size1,batch_first=True)
        self.ch_crf = CRF(self.tagset_size2,batch_first=True)
        self.hidden = self.init_hidden()


    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_feature(self,sentences):
        embeds = self.word_embeds(sentences)
        out , self.hidden = self.lstm(embeds )
        
        
        feats = self.linear(out)

        feats_div = self.hidden2tag1(feats)
        feats_ch = self.hidden2tag2(feats)

        return feats_div , feats_ch        

    def loss(self,sentences , tags_div ,tags_ch ,weight = 0.5):

        feats_div , feats_ch = self._get_feature(sentences)
        crf_loss_div = self.div_crf(feats_div,tags_div , reduction ='mean')

        crf_loss_ch = self.ch_crf(feats_ch , tags_ch , reduction = 'mean')

        return -(crf_loss_div * weight + crf_loss_ch *(1-weight)) , -crf_loss_div , -crf_loss_ch
        #return (crf_loss_ch*crf_loss_div) , -crf_loss_div , -crf_loss_ch

    def forward(self, sentences):  
        

        feats_div , feats_ch = self._get_feature(sentences)

        tags_ch = self.ch_crf.decode(feats_ch)
        tags_div = self.div_crf.decode(feats_div)

        return  tags_div ,tags_ch
    
    def test_sentence(self , sentence):

        sentence = torch.tensor(sentence,dtype = torch.long)
        sentence = sentence.unsqueeze(0)

        tags_div , tags_ch = self.forward(sentence)
        return tags_div[0] , tags_ch[0]




