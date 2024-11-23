import chromadb
import uuid
import sys

from tqdm import tqdm
from embedding.dense import DPRetriever
from utility.general import similarity
from utility.general import splitter

class VectorDB():
    def __init__(self, collection_name, presistent_path) -> None:
        self.collection_name = collection_name
        self.presistent_path = presistent_path
        
        self._init_param()

    def _init_param(self):
        self.client = self._init_client()
        self.collection = self._init_collection()

    def _init_client(self):
        return chromadb.PersistentClient(
            path=self.presistent_path
        )
        
    def _init_collection(self):
        return self.client.get_or_create_collection(self.collection_name)
        
    def add_document(self, documents, chunk_size : int = None, chunk_overlap : int = None):
        retriever = DPRetriever()
        
        for doc in tqdm(documents):
            if chunk_size is not None and chunk_overlap is not None:
                try:
                    sentence_split = splitter(
                        text=doc['context'],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    for sentence in sentence_split:
                        self._document_processing(sentence, retriever)
                except Exception as e:
                    print(e)
            else:
                self._document_processing(doc['context'], retriever)
            
    def _document_processing(self, sentence, retriever):
        doc_embedding = retriever.do_context(
            sentence=sentence,
            model=retriever.context_model(),
            tokenizer=retriever.context_tokenizer()
        )
        
        self.collection.add(
                ids=[str(uuid.uuid4().hex)],
                documents=[sentence],
                embeddings=doc_embedding
            )
        
    def get_document(self, query : str, top_k : int) -> list:
        retriever = DPRetriever()
        
        query_embeddings = retriever.do_context(
            sentence=query,
            model=retriever.context_model(),
            tokenizer=retriever.context_tokenizer()
        )
        
        results = self.collection.query(
            query_embeddings=query_embeddings, 
            n_results=top_k
        )
        
        distance = results['distances'][0]
        document_result = []
        
        for index, document in enumerate(results["documents"][0]):
            candidate_embedding = retriever.do_question(
                model=retriever.question_model(),
                tokenizer=retriever.question_tokenizer(),
                sentence=document
            )    
            
            rank = index + 1
            
            cs = similarity(query=query_embeddings, candidate=candidate_embedding)
            distance_per_index = distance[index]
            
            docset = (rank, distance_per_index, cs, document)
            document_result.append(docset)
            
        return document_result
        
        