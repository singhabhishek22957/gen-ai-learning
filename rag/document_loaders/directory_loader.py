from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate


model = ChatOllama(
    model='gemma2:2b',
    temperature=0.7
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="Write a summary for the following poem - \n {poem}",
    input_variables=['poem']
)
loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()


print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)
