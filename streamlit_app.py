import streamlit as st
import requests
from PIL import Image
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import logging
import json
import time 

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="🤗💬 NCERT AI-Chat Bot")
st.header("NCERT AI-Chat Bot")

# Sidebar
with st.sidebar:
    st.title('🤗💬Sarvam AI Hiring Task')
    st.success('Access to this Gen-AI Powered Chatbot is provided by [Kush](https://www.linkedin.com/in/kush-juvekar/)!', icon='✅')
    st.markdown('⚡ This app is hosted on Lightning AI Studio!')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [ChatMessage(role=MessageRole.ASSISTANT, content="Ask anything about the chapter Sound!")]

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message.role.value):
        st.write(message.content)

# User input
if prompt := st.chat_input("Your question"):
    user_message = ChatMessage(role=MessageRole.USER, content=prompt)
    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        st.write(prompt)

    # Make API call to FastAPI server
    with st.chat_message("assistant"):
        with st.spinner("RAG Answer..."):
            try:
                response = requests.post(
                    "http://localhost:8000/chat",
                    json={
                        "prompt": prompt,
                        "message_history": [
                            {"role": msg.role.value, "content": msg.content}
                            for msg in st.session_state.messages[:-1]
                        ]
                    }
                )
                response.raise_for_status()
                assistant_response = response.json()["response"]
                
                if assistant_response == 'A mindmap is created as per your request.':
                    st.write(assistant_response)
                    with open("mindmap.png", "rb") as file:
                        btn = st.download_button(label="Download Mindmap",data=file,file_name="mindmap.png",mime="image/png",)
                elif assistant_response.startswith('{'):
                    quiz_data = json.loads(assistant_response)
                    st.markdown(f"Question: {quiz_data['question']}")
                    user_choice = st.radio("Options:", quiz_data['choices'],index=None)
                    explanation = quiz_data['explanation']
                    st.write('Answer will be displayed in 5 seconds.')
                    time.sleep(5)
                    st.write("Here's the answer: ",quiz_data['answer'],'\n',f"Here's the explanation: {explanation}")
                else:
                    st.write(assistant_response)
                
                st.session_state.messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response))
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {str(e)}")