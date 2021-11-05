import torch
import torch.nn as nn 
import torch.nn.functional as F
import transformers
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from scipy import spatial
from typing import List
import pandas as pd 
import re
import nltk
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings('ignore')

class SentenceBERTEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_weights = 'sentence-transformers/all-distilroberta-v1'
        self.model = AutoModel.from_pretrained(self.pretrained_weights)
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)     
    def get_bert_embeddings(self, encoded_input):
        with torch.no_grad():
            model_out = self.model(**encoded_input)
        sentence_embedding = self.mean_pooling(model_out, encoded_input['attention_mask'])
        return F.normalize(sentence_embedding, p=2, dim=1)


class FAQBuilder:
    def __init__(self) -> None:
        self.df = pd.read_csv('https://raw.githubusercontent.com/hellohaptik/faq-datasets/master/transformed_quora_dataset/quora_test_10.csv')

        self.MAX_LEN = 16
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.TOKENIZER = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        self.stop_words = set([s for s in stopwords.words('english')]) 
        self.model = SentenceBERTEmbedder()
        self.data_df = self.df.copy()
        self.data_df.loc[:,'clean_questions'] = self.data_df.loc[:,'query'].apply(lambda x:self.text_cleaner(x))
        self.data_df.loc[:,'embeddings'] = self.data_df.loc[:,'clean_questions'].apply(lambda x:self.sentence_infer(x, self.model))

    def text_cleaner(self,text):
        text = text.lower()
        text = re.sub('[^A-Za-z0-9]+', ' ', text)
        text = " ".join([t for t in text.split() if t not in self.stop_words])
        # text = " ".join(text.split())
        return text

    def sentence_tokenizer(self,text):
        encoded_input = self.TOKENIZER(text, padding=True, truncation=True, return_tensors='pt')
        return encoded_input

    def sentence_infer(self,input_txt, model):
        clean_txt = self.text_cleaner(input_txt)
        tokenized_txt = self.sentence_tokenizer(clean_txt)
        out = self.model.get_bert_embeddings(tokenized_txt)
        return out

    def get_FAQ(self,inpt, count=5):
        faq_df = self.data_df
        inpt_emb = self.sentence_infer(inpt, self.model)
        faq_df.loc[:,'similarity'] = faq_df.loc[:,'embeddings'].apply(lambda x: self.cosine_similarity(x, inpt_emb)) 
        # answer = faq_df.iloc[faq_df['similarity'].argmax()].answers
        answer = faq_df.nlargest(count,'similarity')
        return answer

    def cosine_similarity(self,faq_emb, inp_emb):
        return 1 - spatial.distance.cosine(faq_emb, inp_emb)

if __name__ == "__main__":
    fb = FAQBuilder()
    res = fb.get_FAQ('"How can we earn money online in india?"')
    print(res[['query','similarity']])