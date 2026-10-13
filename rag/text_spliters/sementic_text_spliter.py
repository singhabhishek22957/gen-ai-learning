from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings

text = """

# Artificial Intelligence, Machine Learning, and Generative AI

Artificial Intelligence (AI) is a branch of computer science that focuses on creating systems capable of performing tasks that typically require human intelligence. These tasks include problem-solving, decision-making, understanding language, recognizing images, and learning from experience. AI is widely used in healthcare, education, finance, transportation, and software development.

Machine Learning (ML) is a subset of Artificial Intelligence. Instead of explicitly programming every rule, developers train machine learning models using data. The models identify patterns and use those patterns to make predictions. For example, an email spam detection system learns from previously labeled emails to classify new messages as spam or legitimate.

Deep Learning is a specialized area of Machine Learning that uses artificial neural networks with multiple layers. These networks can learn complex patterns from large amounts of data. Deep learning is particularly useful for image recognition, speech processing, natural language understanding, and autonomous driving.

Generative AI focuses on creating new content based on patterns learned from training data. It can generate text, images, audio, video, and computer code. Large Language Models (LLMs) are a type of generative AI model designed to process and generate human language. Examples of LLM applications include chatbots, document summarization, code generation, and question-answering systems.

Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. First, documents are divided into smaller chunks and converted into vector embeddings. These embeddings are stored in a vector database. When a user asks a question, the system retrieves relevant chunks and provides them as context to the language model. This approach helps the model answer questions using information from specific documents rather than relying only on its training data.

Vector embeddings are numerical representations of text that capture semantic information. Texts with similar meanings tend to have embeddings that are close together in vector space. Semantic search uses these representations to retrieve information based on meaning rather than exact keyword matches. For example, a search for "How can I reset my account password?" may retrieve a document titled "Steps to recover your login credentials," even though the wording differs.

Text splitting is an important step in document processing for RAG applications. Large documents are divided into smaller chunks before embedding and indexing. Fixed-size splitters divide text according to character or token limits, while semantic splitters attempt to identify meaningful boundaries by examining changes in the meaning of adjacent text segments. Semantic splitting can help preserve context, although its results depend on the embedding model, similarity threshold, and input content.

A vector database stores and searches vector embeddings efficiently. Popular choices include Qdrant, Chroma, FAISS, and Pinecone. When a query arrives, the database compares its embedding with stored embeddings and returns the most relevant results. These results can then be passed to an LLM to generate a context-aware response.

LangChain is a framework for building applications powered by language models. It provides integrations for document loaders, text splitters, embedding models, vector stores, retrievers, and chat models. Developers can combine these components to build document question-answering systems, conversational assistants, and other AI applications.

Evaluating a RAG system requires more than checking whether its response sounds correct. Important factors include retrieval relevance, answer correctness, faithfulness to the retrieved context, and response latency. A reliable system should acknowledge when the available documents do not contain enough information to answer a question.

"""

embeddings = OllamaEmbeddings(model='nomic-embed-text')

splitters = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)



# res = splitters.split_text(text)
res = splitters.split_text(text)

for i, chunk in enumerate(res, start=1):
    print(f"\n--- Chunk {i} ---\n")
    print(chunk)