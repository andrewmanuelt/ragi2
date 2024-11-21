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
        return DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', device_map='cuda')

    def _ctx_encoder(self):
        return DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base', device_map='cuda')

    def _q_tokenizer(self):
        return DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base', device_map='cuda')

    def _q_encoder(self):
        return DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base', device_map='cuda')

    def context_to_embedding(self, text : str):
        tokenizer = self._ctx_tokenizer()
        input = tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
        encoder = self._ctx_encoder()
        embedding = encoder(**input).pooler_output.detach().numpy()[0]

        return embedding
    
    def question_to_embedding(self, query):
        tokenizer = self._q_tokenizer()
        input = tokenizer(query, return_tensors='pt', truncation=True, max_length=256)
        encoder = self._q_encoder()
        embedding = encoder(**input).pooler_output.detach().numpy()[0]

        return embedding


    