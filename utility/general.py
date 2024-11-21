from sklearn.metrics.pairwise import cosine_similarity

from langchain.text_splitter import RecursiveCharacterTextSplitter

def similarity(query: list, candidate: list):
    return cosine_similarity([query], [candidate])[0][0]

def splitter(text : str, chunk_size : int, chunk_overlap : int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        length_function=len,
        is_separator_regex = False
    )
    
    return splitter.split_text(text)