import os
os.environ['OPENAI_API_KEY'] = "xxx"

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

# vector store that can be initialzied locally. mit license
# might be a better option than pinecone as its free
from langchain_community.vectorstores import FAISS

if __name__ == "__main__":
  pdf_path = "/Users/punkbooster/Projects/langchain/vector-store-in-memory/llm_resoning.pdf"

  # initialize pdf loader
  loader = PyPDFLoader(file_path=pdf_path)

  # load the document
  documents = loader.load()

  # initialzie text splitter
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")

  # split the document
  docs = text_splitter.split_documents(documents=documents)

  # initialzie embeddings
  embedings = OpenAIEmbeddings()

  # takes openai embeddings, converts chunk documents into vectors, takes vectors and stores them in FAISS database.
  # will be stored in RAM locally
  vectorstore = FAISS.from_documents(docs, embedings)

  # persist database in file system locally
  vectorstore.save_local("faith_index")
