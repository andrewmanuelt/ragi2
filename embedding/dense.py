import torch 

from abc import ABC, abstractmethod

from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer

class RetrieverAbstract(ABC):
    def __init__(self) -> None:
        self._embedding_repo = ''
    
    @property
    def embedding_repo(self):
        return self._embedding_repo
    
    @embedding_repo.setter
    def set_embedding_repo(self, embedding_repo):
        self._embedding_repo = embedding_repo
    
    @abstractmethod
    def _ctx_tokenizer(self):
        pass
    
    @abstractmethod
    def _ctx_encoder(self):
        pass
    
    @abstractmethod
    def _q_tokenizer(self):
        pass

    @abstractmethod
    def _q_encoder(self):
        pass
    
    @abstractmethod
    def context_to_embedding(self, text):
        pass
    
    @abstractmethod
    def question_to_embedding(self, document):
        pass
    
class DPRetriever(RetrieverAbstract):
    def _ctx_tokenizer(self):
        return DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

    def _ctx_encoder(self):
        model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return model
        
    def _q_tokenizer(self):
        return DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')

    def _q_encoder(self):
        model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        return model
    
    def context_to_embedding(self, text : str):
        tokenizer = self._ctx_tokenizer()
        input = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input = input.to(device)
        
        encoder = self._ctx_encoder()
        embedding = encoder(**input).pooler_output.cpu().detach().numpy()[0]

        return embedding
    
    def question_to_embedding(self, query):
        tokenizer = self._q_tokenizer()
        input = tokenizer(query, return_tensors='pt', truncation=True, max_length=256)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        input = input.to(device)
        
        encoder = self._q_encoder()
        embedding = encoder(**input).pooler_output.cpu().detach().numpy()[0]

        return embedding


    