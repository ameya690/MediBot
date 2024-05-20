VECTOR_DB_URL = "http://localhost:6333"
VECTOR_DB_NAME = "D:/projects/Medical-RAG-LLM/vector_db"
DATA_DIR = "D:/projects/Medical-RAG-LLM/data/"
EMBEDDINGS = "D:/projects/Medical-RAG-LLM/NeuML/pubmedbert-base-embeddings"
LLM_PATH = "D:/projects/Medical-RAG-LLM/BioMistral-7B.Q4_K_M.gguf"
PROMPT_TEMPLATE = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer. Answer must be detailed and well explained.
Helpful answer:
"""