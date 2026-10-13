import os 

from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama , OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough , RunnableParallel , RunnableLambda 
from langchain_core.output_parsers import StrOutputParser

video_id = "kz_QtbhkM8A"

try:

    # Document Load
    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(
        video_id=video_id,
        languages=["en"]
    ).to_raw_data()
    
    transcript= " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)

    # Text Splitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks  = splitter.create_documents({transcript})

    # print(len(chunks))

    # choose embedding models and make embeddings 

    embeddings = OllamaEmbeddings(
        model='nomic-embed-text:latest'
    )

    vector_store  = FAISS.from_documents(chunks, embedding=embeddings)


    #  retrieval 

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

    # print(retriever)

    # print(context)

    prompt = PromptTemplate(
        template="""
You are a Helpful assistant. 
Answer ONLY from the provided transcript context. If the context is insufficient, Just say you don't know.

{context}
Question: {question}

""",
input_variables=['context', 'question']
    )


  
    # retriever = retriever.invoke(question)
    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    parallel_chain = RunnableParallel({
        'context':retriever | RunnableLambda(format_docs), 
        "question":RunnablePassthrough()
    })
    
    model = ChatOllama(
        model="gemma3:4b",
        temperature=0.7
    )

    parser = StrOutputParser()

    question = "give all project that suggest inside "

    chain = parallel_chain | prompt | model | parser

    res = chain.invoke(question);
    print(res)
    

    # final_prompt = prompt.invoke({"context":context_text,"question":question})

    # print(final_prompt)

    # res = model.invoke(final_prompt)

    # print(res.content)





except TranscriptsDisabled:
    print("No Captions available for this video.")






