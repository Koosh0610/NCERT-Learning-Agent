import logging
import xmltodict
import base64
from graphviz import Digraph

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.schema import QueryBundle, MetadataMode
from llama_index.core.postprocessor import LongContextReorder 
from llama_index.core.retrievers import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

from byaldi import RAGMultiModalModel
from openai import OpenAI

from dotenv import load_dotenv

load_dotenv()


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Colpali Index
RAG = RAGMultiModalModel.from_index("image_index", verbose=0)

# Set up embeddings and LLM/Client
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
llm = Groq(model="llama3-70b-8192", temperature=0)
client = OpenAI(base_url="https://api.groq.com/openai/v1")

# Load index from storage
storage_context = StorageContext.from_defaults(persist_dir="sarvam_ai_index")
index = load_index_from_storage(storage_context)

# Set up retrievers and postprocessor
vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=2)
postprocessor = LongContextReorder()

# Hybrid Retriever class
class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        all_nodes = bm25_nodes + vector_nodes
        all_nodes = postprocessor.postprocess_nodes(nodes=all_nodes, query_bundle=QueryBundle(query))
        return all_nodes

hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)

# Utility function to encode images from OpenAI docs
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Prompt templates
RAG_PROMPT_TEMPLATE = """
You are an artificial intelligence assistant designed to help student answer questions related to a chapter on Sound as part of their curriculum.
The assistant is talkative and provides lots of specific details from the image so that the students can understand concepts as clearly as possible.
Question:
{question}
Context:
{context_str}

Instruction: Based on the image provided, provide a detailed answer to the student. Give more importance to the text.
"""

condense_prompt_template = (
    "Given the following conversation between a user and an AI assistant and a follow up question from user,"
    "rephrase the follow up question to be a standalone question.\n"
    "Chat History:\n"
    "{chat_history}"
    "\nFollow Up Input: {question}"
    "\nStandalone question:")

agent_prompt = """
### Task:
You are an intelligent assistant designed to help users. Your job is to classify the input into one of the following categories based on the user's request:

- Output `0` if the input is a **numerical problem** that needs solving.
- Output `1` if the input is asking to **retrieve context** or information.
- Output `2` if the input is asking to generate or create a **mindmap** for some topic.
- Output `3` if the input is asking to generate a **MCQ** or **quiz**.
- Output `4` if the input is asking for ***more questions similar to the one he asked before**.
- Else, the input will be a friendly dialogue and chat as an assistant.

### Classify the following input:
{input_text}
Please output the appropriate category number ONLY. Reply to the input_text as an assistant if doesn't fall in any category from 0 to 4.
"""

# Quiz generation prompt
QUIZ_PROMPT = """You are a Quiz Master. Your task is to generate a multiple-choice question (MCQ) based on the given context. You will receive a `context_str` as input, and based on that, create an MCQ in the following JSON format:

{{
  "question": "The generated question based on the context.",
  "choices": [
    "Choice A",
    "Choice B",
    "Choice C",
    "Choice D"
  ],
  "answer": "Correct choice",
  "explanation": "Explanation for why the chosen answer is correct."
}}

Here is the context: {context_str}
Here is the user query: {question}

Ensure the `choices` array does not use keys and values. Keep it strictly as a list, and ensure that the question, answer, and explanation are accurate based on the `context_str` provided. **Instruction: Strictly output only the json object and nothing else.**"""

# Function to generate mindmap
def add_graphviz_nodes(dot, node, parent=None):
    if isinstance(node, dict):
        # Retrieve the 'text' attribute if it exists
        text_value = node.get('@text', None)
        
        if text_value:
            if parent:
                # Create a node with the text value and connect to the parent
                new_id = f"{text_value}_{hash(text_value)}"
                dot.node(new_id, text_value)
                dot.edge(parent, new_id)
            else:
                # Root node case (no parent)
                new_id = f"{text_value}_{hash(text_value)}"
                dot.node(new_id, text_value)
                parent = new_id  # The current node becomes the parent
        
        # Iterate over subtopics or other attributes
        for key, value in node.items():
            if key != '@text':
                if isinstance(value, list):
                    # Multiple subtopics or children
                    for subtopic in value:
                        add_graphviz_nodes(dot, subtopic, parent)
                else:
                    # Single subtopic
                    add_graphviz_nodes(dot, value, parent)
                    
    elif isinstance(node, list):
        for item in node:
            add_graphviz_nodes(dot, item, parent)


def old_add_graphviz_nodes(dot, node, parent=None):
    if isinstance(node, dict):
        for key, value in node.items():
            dot.node(key, key)
            if parent:
                dot.edge(parent, key)
            add_graphviz_nodes(dot, value, key)
    elif isinstance(node, list):
        for item in node:
            add_graphviz_nodes(dot, item, parent)
    else:
        dot.node(node, node)
        if parent:
            dot.edge(parent, node)

def generate_mindmap(condensed_question):
    nodes = hybrid_retriever.retrieve(str(condensed_question))
    context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes])
    xml_string = llm.complete(f"""You are a high school teacher loved by all students. Your job is to write xml code for the topic requested by the student and the provided context. Here is the question: {condensed_question}. An here is the context: {context_str}. \*\*Instruction: You'll only give the xml code and nothing else. It should begin with: <?xml version="1.0" encoding="UTF-8"?>\*\*""")
    
    with open('output.xml', 'w') as xml_file:
        xml_file.write(xml_string.text)
    
    with open('output.xml', 'r') as xml_file:
        xml_content = xml_file.read()
    xml_dict = xmltodict.parse(xml_content)

    # Create a Digraph object
    dot = Digraph(format='png')
    dot.attr(size="10,10!")
    dot.attr(dpi="300")
    try:
        add_graphviz_nodes(dot, xml_dict)
    except Exception as e:
        logging.info(f"An unexpected error occurred: {str(e)}")
    dot.render('mindmap', format='png')
    
    return ""

# Function to get quiz data
def get_quiz_data(condensed_question):
    nodes = hybrid_retriever.retrieve(str(condensed_question))
    context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes])
    response = llm.complete(QUIZ_PROMPT.format(context_str=context_str, question=condensed_question))
    return response.text

def get_rag_answer(condensed_question):
    nodes = hybrid_retriever.retrieve(str(condensed_question))
    context_str = "\n\n".join([n.node.get_content(metadata_mode=MetadataMode.LLM).strip() for n in nodes])
    results = RAG.search(f"{condensed_question}",k=1)
    image_index = results[0]['page_num']
    if image_index < 10:
        base64_image = encode_image(f"document_images/7261ff67-0f02-4a11-a8f8-60264af5c3cf-0{image_index}.jpg")
    else:
        base64_image = encode_image(f"document_images/7261ff67-0f02-4a11-a8f8-60264af5c3cf-{image_index}.jpg")
    messages=[{
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": RAG_PROMPT_TEMPLATE.format(question=condensed_question,context_str=context_str),
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
          },
        },
      ],
    }]
    model="llava-v1.5-7b-4096-preview"
    return client.chat.completions.create(messages=messages,model=model).choices[0].message.content