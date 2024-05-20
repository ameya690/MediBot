from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

import settings

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
QA_DATA_PATH = 'medical_meadow_small.json'  # Path to your JSON file with question-answering data

embeddings = SentenceTransformerEmbeddings(model_name=settings.EMBEDDINGS)

print(embeddings)

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def load_qa_data(file_path):
    with open(file_path, 'r') as file:
        qa_data = json.load(file)
    documents = []
    for item in qa_data:
        documents.append(Document(
            page_content=item['input'] + "\n" + item['output'],
            metadata={'instruction': item['instruction']}
        ))
    return documents

def create_vector_db():
    # Load PDF documents
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    pdf_documents = loader.load()
    
    # Load QA data
    qa_documents = load_qa_data(QA_DATA_PATH)
    
    # Combine both sets of documents
    documents = pdf_documents + qa_documents
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == "__main__":
    create_vector_db()
