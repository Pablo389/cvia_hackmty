#Librerias para las cadenas y memoria
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

#Librerias para el embedding y retrieval
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.chains import RetrievalQA

#Librerias para los prompts de chat
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import os
import streamlit
from PyPDF2 import PdfReader

#Constante api open ai key

def vector_qa(pdf_docs):
    text = ""
    contador = 0
    for pdf in pdf_docs:
        contador += 1
        text += "INICIO DEL CV {contador}"
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            
            text += page.extract_text()
        
        text += "FIN DEL CV"
            
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    vectordb = FAISS.from_texts(texts=chunks, embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

    #qa marca es la herramienta con la que haremos retrieval de documentos, retorna texto
    qa_marca = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY,model_name = "gpt-3.5-turbo-16k"), chain_type= "stuff", retriever = vectordb.as_retriever())
    
    return qa_marca

def chat_prompt():
    #Template del prompt que va a ser enviado al modelo de texto para que me regrese el mejor mensaje
    template="""Eres un chatbot especializado en revisar y comprender CV, tu trabajo es ayudar a los reclutadores sobre los CV que se te presenta. Este es el historial de mensajes:
    {chat_history}
    Y esta es la información que recabaste para poder responder la pregunta:
    {qa_answer}

    A continuacion esta el mensaje del responsable de recursos humanos:
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    cht_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    return cht_prompt

def chat_memory(qa_marca):
    #Inicializar la memoria
    memory = ConversationBufferMemory(memory_key="chat_history",input_key="text", return_messages=True)
    chat=ChatOpenAI(openai_api_key=OPENAI_API_KEY,temperature=0, model_name="gpt-3.5-turbo-16k")
    chain = LLMChain(llm=chat, prompt=chat_prompt(), memory=memory, verbose=True)
    while True:
        pregunta = input()
        if pregunta == "exit":
            break
        respuesta_impulse = qa_marca.run(pregunta)
        resp_ia = chain.run(question = pregunta, text = pregunta, qa_answer = respuesta_impulse)
        print(resp_ia)


def chat(qa_marca):
    while True:
        pregunta = input()
        if pregunta == "exit":
            break
        print(qa_marca.run(pregunta))

        
if __name__ == "__main__":
    pdfs = [element for element in os.listdir("PDFS")]
    qa = vector_qa(pdfs=pdfs)  
    chat(qa_marca=qa)