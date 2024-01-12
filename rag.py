# Import necessary libraries
import streamlit as st
from streamlit_chat import message
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from langchain.llms import WatsonxLLM
from langchain.chains import RetrievalQA

# Set up the credentials and project_id
credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "09naIbU_40U9bzS3yw2ZiRkDY1va1k5FCwei05DVa3k9"
}

try:
    project_id = "0ece3ec4-a895-4b50-aca2-5794f884c3bb"
except KeyError:
    project_id = st.text_input("Enter your project_id:")

model_id = ModelTypes.FLAN_UL2

embeddings = HuggingFaceEmbeddings()

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 100,
    GenParams.STOP_SEQUENCES: ["<|endoftext|>"]
}

# Load the model of choice
def load_llm(parameters):
    watsonx_granite = WatsonxLLM(
        model_id=model_id.value,
        url=credentials.get("url"),
        apikey=credentials.get("apikey"),
        project_id=project_id,
        params=parameters
    )
    return watsonx_granite

# Set the title for the Streamlit app
st.title("Welcome to BigCash- 🦜🦙")

# Exception handling for loading LLM
try:
    llm = load_llm(parameters)
except Exception as e:
    st.error(f"Error loading LLM: {e}")
    st.stop()

persist_directory='C:/Users/PriKumar/AIML/IBM POC/storage/bgc'
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Create a conversational chain
chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordb.as_retriever())

# Function for conversational chat
def conversational_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Initialize chat history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Initialize messages
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello ! how can i help you " + " 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]

# User input form
with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_input("Query:", placeholder="Talk to Bigcash Support 👉 (:")
    submit_button = st.form_submit_button(label='Send')

if submit_button and user_input:
    output = conversational_chat(user_input)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

# Display chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])):
        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
        message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")
