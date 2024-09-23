#-------------------------------------------------#
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
#-------------------------------------------------#
from llama_index.core.schema import MetadataMode
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.base import PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.core.base.llms.generic_utils import messages_to_history_str
from byaldi import RAGMultiModalModel
from openai import OpenAI
from utils_new import condense_prompt_template,agent_prompt, generate_mindmap, get_quiz_data, get_rag_answer
import logging
#-------------------------------------------------#
from dotenv import load_dotenv
load_dotenv()

#Configer Logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#Initialiazing FastAPI app
app = FastAPI()

# Initialize your models and clients here
llm = Groq(model="llama3-70b-8192", temperature=0,)


#Defining Pydantic Models
class ChatRequest(BaseModel):
    prompt: str
    message_history: List[ChatMessage]

    class Config:
        arbitrary_types_allowed = True

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        response = ncert_agent(prompt=request.prompt, message_history=request.message_history)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# NCERT Agent
def ncert_agent(prompt,message_history):
  action = llm.complete(agent_prompt.format(input_text=prompt))
  if action.text == '0': # Numerical Solving
    return llm.complete(prompt).text
  if action.text == '1': # RAG
    condensed_question = llm.predict(PromptTemplate(condense_prompt_template),chat_history=messages_to_history_str(message_history),question=prompt)
    message_history.append(ChatMessage(role=MessageRole.USER,content=condensed_question))
    return get_rag_answer(condensed_question)
  if action.text == '2': #Mindmap Generation
    condensed_question = llm.predict(PromptTemplate(condense_prompt_template),chat_history=messages_to_history_str(message_history),question=prompt)
    path_to_mindmap = generate_mindmap(condensed_question=condensed_question)
    response = 'A mindmap is created as per your request.'
    return response
  if action.text == '3': # MCQ Generation
    condensed_question = llm.predict(PromptTemplate(condense_prompt_template),chat_history=messages_to_history_str(message_history),question=prompt)
    return get_quiz_data(condensed_question)
  if action.text == '4': # Similar question generation
    condensed_question = llm.predict(PromptTemplate(condense_prompt_template),chat_history=messages_to_history_str(message_history),question=prompt)
    response = llm.complete(f"You're a exam paper setter. You are supposed to generate two advanced questions on the concepts which are being used in the following question. DO NOT change the concept or formulas required. You are only onlyy allowed to make the porblem more difficult to understand or computationally harder to solve via pen and paper. Here's the question: {condensed_question}.").text
    return response
  else:
    return action.text 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
