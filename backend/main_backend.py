from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel 
from rag_chain import load_rag_chain 
from fastapi.middleware.cors import CORSMiddleware 

# akses dari frontend Streamlit/localhost
app_fastApi = FastAPI() 
app_fastApi.add_middleware(
    CORSMiddleware, 
    allow_origins= ['*'], 
    allow_credentials= True, 
    allow_methods= ["*"], 
    allow_headers =["*"], 
)

# loading chain 
rag_chain = load_rag_chain() 

# request body 
class Query_Request(BaseModel): 
    input: str 

# Response body 
class Query_Response(BaseModel): 
    answer: str 
    sources: list[str] 


@app_fastApi.get('/testStatus')
def root(): 
    test_status_result ={
        'status': 'okay', 
        "msg": 'API is running. use POST /ask'
    }
    return test_status_result

@app_fastApi.post('/ask', response_model=Query_Response) 
def ask_question(request: Query_Request): 
    try: 
        result = rag_chain.invoke({'query': request.input}) 
        answer = result['result'] 
        sources = [] 

        for doc in result.get('source_documents', []): 
            metadata = doc.metadata 
            if 'source' in metadata: 
                sources.append(metadata['source']) 
            elif 'file_path' in metadata: 
                sources.extend(metadata['file_path']) 

        return Query_Response(answer=answer, sources=sources)
    
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))