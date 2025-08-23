import streamlit as st 
import requests 

st.set_page_config(page_title='RAG Chatbot', page_icon='🤖') 

st.title('RAG Internal Chatbot') 
st.caption('powered by FastAPI + LangChain + DeepSeek') 

# initialize chat history 
if 'messages' not in st.session_state: 
    st.session_state.messages= [] 

# show chat history 
for msg in st.session_state.messages: 
    with st.chat_message(msg['role']): 
        st.markdown(msg['content']) 

# User input 
if prompt := st.chat_input('Do u wanna ask something about the docs? \n'): 
    # add user message to chat 
    st.session_state.messages.append({
        'role': 'user', 
        'content': prompt
    }) 
    with st.chat_message('user'): 
        st.markdown(prompt) 

    # send to FastAPI Backend --------------- 
    try: 
        with st.spinner('trying to answer...'): 
            res = requests.post('http://localhost:8000/ask', json={'input': prompt}) 
            data = res.json() 
            answer = data.get('answer', 'Failed to answer :(') 
            sources = data.get('sources', []) 
            answer_with_sources = f"{answer} \n\n**Sources:** \n" + '\n'.join(f"- `{s}`" for s in sources) 


            # add assistant response 
        st.session_state.messages.append({
            'role': 'assistant', 
            'content': answer_with_sources,
        })
        with st.chat_message("assistant"): 
            st.markdown(answer_with_sources)

    except Exception as e: 
        error_message = f'Failed to calling the FastAPI: {e}' 
        st.session_state.messages.append({
            'role': 'assistant', 
            'content': error_message,
        })
        with st.chat_message('assistant'): 
            st.error(error_message)