from langchain_huggingface import HuggingFaceEmbeddings
import os 


os.environ["HF_HOME"] = "E:/Huggingface_embedding"
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

test = [" Delhi is the capital of India", " Mumbai is the capital of India", " New Delhi is the capital of India"]

# vector = embedding.embed_query(test)
vector = embedding.embed_documents(test)

print(vector)
