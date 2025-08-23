'''
    MEMBUAT & MENYIMPAN VECTORSTORE

    fungsi script ini adalah: 
    - load the documents 
    - split jadi chunk otomatis (pake RecursiveCharacterTextSplitter) 
    - embed dengan hugging face 
    - simpan ke chromadb (persis dir)
'''

from langchain_community.document_loaders import TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os 
import torch
from dotenv import *
from backend.config import *

def load_documents(data_dir): 
    # LOAD Process  
    # Loading process dibawah ini cara kerjanya adalah dengan membaca dan mengumpulkan 
    # semua dokumen/file, lalu menjadikannya jadi 1 Vector_store Chromadb di 1 Folder
    all_docs = [] 
    for filename in os.listdir(data_dir): 
        if filename.endswith('.txt'): 
            loader = TextLoader(os.path.join(DATA_DIR, filename), encoding='utf-8') 
            docs = loader.load() 
            all_docs.extend(docs) 
    return all_docs


def ingest(): 
    print('Loading documents...') 
    docs = load_documents(DATA_DIR) 

    print('Splitting documents...') 

    # SPLIT Process 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size= CHUNK_SIZE, 
        chunk_overlap= CHUNK_OVERLAP,
    )
    chunks_process = text_splitter.split_documents(docs) 

    # EMBED Process 
    print('Embedding and storing to ChromaDB...')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = HuggingFaceEmbeddings(
        model_name= EMBEDDING_MODEL, 
        model_kwargs = {'device': device}
    ) 

    # BUILD & SAVE vector store 
    vectordb = Chroma.from_documents(
        documents=chunks_process, 
        embedding=embedding_model, 
        persist_directory=DB_DIR,
    )
    
    print(f'Vectorstore created and already saved in: {DB_DIR}')


if __name__ == '__main__': 
    ingest()