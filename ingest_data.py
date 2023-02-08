from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceHubEmbeddings
import pickle
import os
from dotenv import load_dotenv
load_dotenv()

huggingfacehub_api_token = os.environ.get("huggingfacehub_api_token") 

# Load Data
loader = UnstructuredFileLoader("state_of_the_union.txt")
raw_documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(raw_documents)


# Load Data to vectorstore
embeddings = HuggingFaceHubEmbeddings(huggingfacehub_api_token=huggingfacehub_api_token)
vectorstore = FAISS.from_documents(documents, embeddings)


# Save vectorstore
with open("vectorstore.pkl", "wb") as f:
    pickle.dump(vectorstore, f)
