import os 
import sys 
import json
import pandas as pd

from utility.db import VectorDB
from utility.document import DocumentLoader

class RetrieverTuning():
    def __init__(self, collection_name, presistent_path, train_path, test_path):
        self.collection_name = collection_name
        self.presistent_path = presistent_path
        self.train_dataset_path = train_path
        self.test_dataset_path = test_path
        self.prep_dataset_path = None
        
    def parameter(self):
        param = {
            'chunk_size': [300, 500, 700, 1000],
            'chunk_overlap': [0, 30, 50, 60], 
            'top_k': [3, 5, 10]
        }
        
        return param['chunk_size'], param['chunk_overlap'], param['top_k']
    
    def _prepare_train_dataset(self, chunk_size, chunk_overlap, top_k):
        if not os.path.exists(self.train_dataset_path):
            raise FileNotFoundError(f'Path {self.train_dataset_path} is not exist !')

        try:
            loader = DocumentLoader(dataset_path=self.train_dataset_path)
            document = loader.documents()
            
            db = VectorDB(
                collection_name=f"{self.collection_name}_{chunk_size}_{chunk_overlap}_{top_k}", 
                presistent_path=f"{self.presistent_path}/{chunk_size}_{chunk_overlap}_{top_k}"
            )
            
            db.add_document(
                documents=document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        except Exception as e:
            print(e)
            
    def _load_test_dataset(self):
        if not os.path.exists(self.test_dataset_path):
            raise FileNotFoundError(f'Path {self.test_dataset_path} is not exist !')
        
        test = None
        with open(self.test_dataset_path) as f:
            test = json.load(f)
        
        collection = {
            'data': []
        }
        
        for row in test:
            collection['data'].append(row['question'])
        
        with open(f'./hyperparameter/{self.collection_name}.json') as f:
            json.dump(collection, f)    
        
        return collection['data']
        
    def tuning(self):
        df = pd.DataFrame(columns=['rank', 'distance', 'cosine_sim', 'text', 'chunk_size', 'chunk_overlap', 'top_k', 'collection'])
        chunk_sizes, chunk_overlaps, top_ks = self.parameter()
        
        for chunk_size in chunk_sizes:
            for chunk_overlap in chunk_overlaps:
                for top_k in top_ks:
                    df = self._repetition(df, chunk_size, chunk_overlap, top_k)
        
        df.to_csv('./hyperparameter/result.csv')
        
    def _repetition(self, df, chunk_size : int, chunk_overlap : int, top_k : int) -> pd.DataFrame:
        self._prepare_train_dataset(chunk_size, chunk_overlap, top_k)
        print('1')
        sys.exit(1)
        test = self._load_test_dataset()
        
        new_collection_name = f"{self.collection_name}_{chunk_size}_{chunk_overlap}_{top_k}"
        
        db = VectorDB(
            collection_name=new_collection_name, 
            presistent_path=f"{self.presistent_path}/{chunk_size}_{chunk_overlap}_{top_k}"
        )    
        
        db.collection_is_exist(self.collection_name)
        
        for row in test:
            result = db.get_document(row, top_k)
            df = self._reformating_tuning(df, result, chunk_size, chunk_overlap, top_k, new_collection_name)

        return df
        
    def _reformating_tuning(self, df : pd.DataFrame, result : list, chunk_size : int, 
                            chunk_overlap : int, top_k : int, new_collection_name : str) -> pd.DataFrame:    
        for row in result:
            row = row + (chunk_size,)
            row = row + (chunk_overlap,)
            row = row + (top_k,)
            row = row + (new_collection_name,)

            df.loc[len(df)] = row
            
        return df
        
        
        
        