import pandas as pd

from utility.db import VectorDB
from utility.document import DocumentLoader
from utility.general import check_device

from hyperparameter.tuning import RetrieverTuning

def documents():
    loader = DocumentLoader(dataset_path='./dataset/single/single_train.json')
    return loader.documents()
    
def main(): 
    db = VectorDB(
        collection_name='single', 
        presistent_path='./database/single'
    )
    
    db.add_document(
        documents=documents(),
    )
    
    # result = db.get_document(
    #     query="Di mana terletak di provinsi trento ?",
    #     top_k=5,
    # )
    
    # print(result)
    
if __name__ == '__main__':
    check_device()
    main()