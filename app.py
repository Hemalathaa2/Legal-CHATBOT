import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
st.set_page_config(
    page_title="IPC Legal Chatbot",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set up environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),
               model_name="llama3-70b-8192")
groq_api_key=os.getenv("GROQ_API_KEY")
# Streamlit UI setup
st.title(" Indian Penal Code ChatBot")
st.markdown("🔍 Ask any question related to the Indian Penal Code and get instant answers.")
st.image("legal.jpg", use_container_width=True)

st.markdown("""<style>
    .stApp {
        background-image: url('legal.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position:center;
        position:relative;
        min-height: 100vh;
    }
     .overlay {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(255, 255, 255, 0.5);
        pointer-events: none; /* So it doesn't block clicks */
        z-index: 0;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.7); /* translucent dark */
        color: white;
        position: relative;
        z-index: 1;
        padding: 2rem;
        border-radius: 10px;
        max-width: 900px;
        margin: 2rem auto;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(5px);
        -webkit-backdrop-filter: blur(5px);
    }
    h1 {
        text-align: center;
        font-weight: 900;
        font-size: 3rem;
        margin-bottom: 1rem;
        color: #111111;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2); 
    }
    
    <div style='margin-top: 20px; font-weight: bold; font-size: 20px; color: darkred;'>
ANSWER:
</div></style>
""", unsafe_allow_html=True)

# Reset conversation function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Define the prompt template
prompt_template = """
<s>[INST]This is a chat template and As a legal chat bot , your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to the Indian Penal Code.
CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
</s>[INST]
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Initialize the LLM


# Set up the QA chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message.get("role")):
        st.write(message.get("content"))

# Input prompt
input_prompt = st.chat_input("Say something")

if input_prompt:
    with st.chat_message("user"):
        st.write(input_prompt)

    st.session_state.messages.append({"role": "user", "content": input_prompt})

    with st.chat_message("assistant"):
        with st.status("Thinking 💡...", expanded=True):
            result = qa.invoke({"question": input_prompt})
            message_placeholder = st.empty()
            full_response = "\n\n\n"

            # Print the result dictionary to inspect its structure
            #st.write(result)

            #for chunk in result["answer"]:
            full_response = result["answer"]
            time.sleep(0.02)
            message_placeholder.markdown(full_response + " ▌")

            # Print the answer
            #st.write(result["answer"])

        st.button('Reset All Chat 🗑️', on_click=reset_conversation)
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
