from langchain_huggingface import HuggingFaceEmbeddings

from sklearn.metrics.pairwise import cosine_similarity
import os 
import numpy as np 




os.environ["HF_HOME"] = "E:/Huggingface_embedding"
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


documents = [
    """
    FastAPI is a modern Python web framework used to build APIs quickly.
    It is based on Starlette for web handling and Pydantic for data validation.
    FastAPI supports asynchronous programming, dependency injection, automatic
    OpenAPI documentation, request validation, response models, middleware,
    authentication, and background processing. It is commonly used for
    microservices, machine learning APIs, and backend applications where
    performance and developer productivity are important.
    """,

    """
    Redis is an in-memory data structure store commonly used as a cache,
    message broker, session store, and job queue. Redis supports strings,
    lists, sets, sorted sets, hashes, streams, and other data structures.
    In a distributed application, Redis can be placed between an API server
    and background workers. The API can push a job into a Redis queue while
    a worker consumes the job asynchronously and performs long-running work.
    """,

    """
    Retrieval Augmented Generation, commonly called RAG, combines information
    retrieval with a large language model. Documents are first loaded,
    divided into smaller chunks, converted into vector embeddings, and stored
    inside a vector database. When a user asks a question, the question is
    converted into an embedding and compared with stored vectors using
    semantic similarity. The most relevant chunks are retrieved and provided
    to the language model as context for generating an answer.
    """,

    """
    Docker allows applications and their dependencies to run inside isolated
    containers. Docker Compose can be used to run multiple services such as
    FastAPI, MongoDB, Redis, workers, and vector databases together. Services
    communicate using Docker's internal network and service names. Persistent
    data can be stored using Docker volumes. For example, a worker can clone
    a Git repository into a shared volume, process the files, generate
    embeddings, store the vectors in a vector database, and then remove the
    temporary repository.
    """,

    """
    Semantic search attempts to find information based on meaning rather than
    exact keyword matching. An embedding model converts text into a numerical
    vector representing its semantic characteristics. Similar concepts tend
    to produce vectors that are close to each other in vector space. For
    example, a search for 'How can I run asynchronous tasks in Python?' may
    retrieve a document discussing FastAPI background workers even if the
    document never contains the exact phrase used in the query. Vector
    databases such as Qdrant, Pinecone, Weaviate, and Milvus can efficiently
    search these embeddings.
    """
]

test = "What is the purpose of converting documents into vectors?"

doc_embedding = embedding.embed_documents(documents)
vector = embedding.embed_query(test)

scores  = cosine_similarity([vector], doc_embedding)[0]
index , score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]
print(test)
print(documents[index])
print( "Similarity Score : ", score)