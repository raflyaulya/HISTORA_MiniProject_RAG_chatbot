"""
    Sebelum melakukan atau menggunakan RAG query,
    hal pertama yg mesti dilakukan adalah harus do the Chunking step terlebih dahulu
      
    karena data2 atau pun dokumen2 yg dimiliki, system mesti membaca, menganalisa & menyimpannya 
    kedalam memory, sehingga user bisa melakukan query

    NOTE!! 
    bisa diperbaiki lagi dikit2, misal dokumennya yg bener2 valid & akurat, 
    lalu perbanyak dokumen supaya result nya makin bagus
      
"""

# from langchain_community.vectorstores import Chroma 
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain.chains.retrieval_qa.base import RetrievalQA  
import torch
import sys
import os
sys.path.append(os.path.dirname(__file__))

from backend.setup import llm_deepseek
from backend.config import BASE_DIR, CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, DB_DIR, DEEPSEEK_API, EMBEDDING_MODEL



def load_rag_chain(): 
    # print('Loading chroma vectorstore...')
    # Load Vectorstore 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embedding_model = HuggingFaceEmbeddings(
        model_name= EMBEDDING_MODEL, 
        model_kwargs = {'device': device}
        ) 
    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embedding_model
        )

    # Retrieve 
    retriever = vectordb.as_retriever()

    # print('Loading LLM (Deepseek)...')
    #  LLM model Deepseek 
    llm_model= llm_deepseek()

    # Building chain
    # print('Building RAG Chain...') 
    chain= RetrievalQA.from_chain_type( 
        llm=llm_model, 
        retriever= retriever, 
        return_source_documents = True
    )

    return chain


def border_chat(): 
    borders = '=' *50 
    return borders 


# TESTING PART here below 
if __name__ == '__main__': 
    chain  = load_rag_chain() 

    print(border_chat())
    ask_theAi = input('\nDo u wanna ask something? \n')
    result = chain.invoke(
        {'query': ask_theAi}
    )
    # the Output 
    print(result['result'])


# Contoh Pertanyaan Buat Ngetes:
# "Apa itu Perang Punik?"
# "Apa yang terjadi dalam Peristiwa Rengasdengklok?"
# "Bagaimana pengaruh Revolusi Industri terhadap pertumbuhan kota?"
# "Kapan Kekaisaran Romawi runtuh dan kenapa?"