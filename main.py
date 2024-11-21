import pandas as pd

from utility.db import VectorDB
from utility.document import DocumentLoader

from hyperparameter.tuning import RetrieverTuning

def documents():
    loader = DocumentLoader(dataset_path='./dataset/dummy.json')
    return loader.documents()
    
def main(): 
    # db = VectorDB(
    #     collection_name='dummy', 
    #     presistent_path='./database/dummy'
    # )
    
    # db.add_document(
    #     documents=documents(),
    #     chunk_size=100,
    #     chunk_overlap=10
    # )
    
    # query = "Di mana terletak di provinsi trento ?"
    # result = db.get_document(
    #     query=query,
    #     top_k=5
    # )
    # print(result)
    
    # tuner = RetrieverTuning(
    #     collection_name='dummy', 
    #     presistent_path='./database/dummy', 
    #     test_path='./dataset/dummy_test.json',
    #     train_path='./dataset/dummy.json'
    # )
    # tuner.tuning()
    
    
    
    
if __name__ == '__main__':
    main()