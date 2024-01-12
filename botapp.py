from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from langchain.llms import WatsonxLLM
from langchain.chains import RetrievalQA
from flask import Flask, jsonify, request, render_template
from waitress import serve


from googletrans import Translator
from flask_cors import CORS




translator = Translator()

def detect_and_translate(sentence):
    translator = Translator()
    # Detect the language of the input sentence
    detected_language = translator.detect(sentence).lang
 
    # Translate the sentence to English
    
    translated_sentence = translator.translate(sentence, src=detected_language, dest='en').text
 
    return detected_language, translated_sentence

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

model_id2 = ModelTypes.GRANITE_13B_CHAT_V2
chat = WatsonxLLM(model_id=model_id2.value,url=credentials.get("url"),apikey=credentials.get("apikey"),project_id=project_id,params=parameters)

try:
    llm = load_llm(parameters)
except Exception as e:
    st.error(f"Error loading LLM: {e}")
    
vectordbFAQ_BGC = Chroma(persist_directory="C:/Users/PriKumar/AIML/IBM POC/storage/faq", embedding_function=embeddings)
# chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectordbFAQ_BGC.as_retriever())
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordbFAQ_BGC.as_retriever())
qa2 = RetrievalQA.from_chain_type(llm=chat, chain_type="stuff", retriever=vectordbFAQ_BGC.as_retriever())

app = Flask(__name__)
CORS(app)

@app.route('/rag_chat', methods=['POST'])
def get_record():
    try:
        json_data = request.json
        query = json_data.get('message', '')
        
        valid_greetings = ["hi", "hello", "hey", "morning", "afternoon", "evening"]
        
        if query.lower() in valid_greetings:
            return jsonify({"result": "Hello, welcome to bigcash! How can I assist you"})
        else:
            detected_language, query = detect_and_translate(query)
            result = chain.run(query)
            
            if result in ['Unhelpful', "I don't know", "Unanswerable"]:
                result2 = qa2.run(query)
                result = f"I am not aware of it, searching on the internet. Here is what I found: {result2}"
                result = translator.translate(result, src='en', dest=detected_language).text
                print(detected_language)
                return jsonify({"result": result})
            else:
                print(detected_language)
                result = translator.translate(result, src='en', dest=detected_language).text
                return jsonify({"result": result})
    except Exception as e:
        error_message = f"Error occurred: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500  # Return a 500 Internal Server Error status code

if __name__ == '__main__':
    serve(app, port="9090")
