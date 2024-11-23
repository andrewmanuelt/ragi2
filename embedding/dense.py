import torch 

from transformers import AutoModel, AutoTokenizer

class DPRetriever():    
    def _device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        
    def context_model(self):
        device = self._device()
        
        model = AutoModel.from_pretrained('sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base')
        model = model.to(device)
        return model 
    
    def context_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base')
        return tokenizer
    
    def do_context(self, sentence, model, tokenizer):
        device = self._device()
        
        encoded = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
        
        with torch.no_grad():
            model_output = model(**encoded)
                
        embedding = model_output[0][:,0]
        embedding = embedding.cpu().tolist()
        
        return embedding
    
    
    def question_model(self):
        device = self._device()
        
        model = AutoModel.from_pretrained('sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
        model = model.to(device)
        return model 
    
    def question_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
        return tokenizer
    
    def do_question(self, sentence, model, tokenizer):
        device = self._device()
        
        encoded = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
        encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
        
        with torch.no_grad():
            model_output = model(**encoded)
        
        embedding = model_output[0][:,0]
        embedding = embedding.cpu().tolist()
        
        return embedding