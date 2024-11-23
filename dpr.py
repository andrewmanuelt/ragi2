import sys 
import faiss 

from tqdm import tqdm
from uuid import uuid4
 
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from utility.document import DocumentLoader

loader = DocumentLoader(
    dataset_path='./dataset/single/single_train.json'
)
documents = loader.documents()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base")
# qembeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/facebook-dpr-question_encoder-single-nq-base")

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore= InMemoryDocstore(),
    index_to_docstore_id={}
)

for doc in tqdm(documents):
    d = Document(
        page_content=doc['context'], 
        metadata={
            "q": doc['question'],
            "a": doc["answer"]
        }
    )
    
    id = str(uuid4())
    
    vector_store.add_documents(documents=[d], ids=[id])
vector_store.save_local(folder_path='./experimental/index5', index_name='index5')


# work !
# d1 = Document(page_content="Grauno adalah komune yang terletak di Provinsi Trento, Italia.", metadata={"q": "Grauno", "a": "Di mana terletak di provinsi trento ?"})
# d2 = Document(page_content="Grauno adalah komune yang terletak di Provinsi Trento, Italia.", metadata={"q": "Frazione apa terletak di provinsi trento ?", "a": "Grauno"})
# d3 = Document(page_content="Iriga City adalah kota yang terletak di provinsi Camarines Sur, Filipina.Pada tahun 2007, kota ini memiliki populasi sebesar 97.983 jiwa atau 17.061 rumah tangga.Pembagian wilayahSecara politis Iriga City terbagi menjadi 36 barangay, yaitu:Pranala luarOfficial Website of the City of Iriga Diarsipkan 2020-03-19 di Wayback Machine.FPJ's Blog Skulakog by H. Frank V. Penones, Jr.Iriga Historical Timeline2007 NSCB information Diarsipkan 2016-04-03 di Wayback Machine.Philippine Standard Geographic Code Diarsipkan 2012-04-13 di Wayback Machine.2000 Philippine Census Information2007 Philippine Census Information Diarsipkan 2009-03-02 di Wayback Machine.Iriga City, WorldIriga City, Philippines: Great People, Great DestinationNCC Website for Iriga City Diarsipkan 2007-03-11 di Wayback Machine.News from IrigaUniversity of Saint Anthony", metadata={"q": "Iriga terletak di daftar provinsi di filipina apa ?", "a": "Camarines Sur"})
# d4 = Document(page_content="Deli Serdang (abjad Jawi:  dly srdnGH) adalah sebuah kabupaten yang berada di provinsi Sumatra Utara, Indonesia.", metadata={"q": "Kabupaten apa kabupaten sumatra utara ?", "a": "Deli Serdang"})

# vector_store.add_documents(documents=[d1], ids=['1'])
# vector_store.add_documents(documents=[d2], ids=['2'])
# vector_store.add_documents(documents=[d3, d4], ids=['3', '4'])

# docs = [d1, d2, d3, d4]
# idx = ['1', '2', '3', '4']

# for i, doc in enumerate(docs):
#     vector_store.add_documents(documents=[doc], ids=[i])

# vector_store.save_local(folder_path='./experimental/index1', index_name='index1')

vector_store = FAISS.load_local(
    folder_path='./experimental/index5', 
    index_name='index5',
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

results = vector_store.similarity_search(query="pluto", k=4)
for doc in results:
    print(f"* {doc.page_content} [{doc.metadata}]")
    

