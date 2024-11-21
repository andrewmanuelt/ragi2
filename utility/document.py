import json 

from langchain_community.document_loaders import JSONLoader

class DocumentLoader():
    def __init__(self, dataset_path):
        self.path = dataset_path
        
    def _load(self):
        loader = JSONLoader(
            file_path=self.path,
            jq_schema=".[]",
            metadata_func=self._metadata_processing,
            text_content=False
        )
        
        return loader.load()
    
    def _metadata_processing(self, record, metadata):
        metadata['answer'] = record.get('answer')
        metadata['question'] = record.get('question')
            
        return metadata
    
    def documents(self):
        collection = []
        
        documents = self._load()
        for document in documents:
            document_object = json.loads(document.page_content)    
            
            collection.append(document_object)

        return collection