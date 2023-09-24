from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone, FAISS

loader = PyPDFLoader("PDFS\Resume_MongoDB.pdf")
pages = loader.load_and_split()

print(pages[0])

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())