from langchain_community.document_loaders import PyPDFLoader
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
loader = PyPDFLoader('pdf.pdf')

docs = loader.load()

print(type(docs))
print(len(docs))

print(type(docs[0]))
# print(docs[0])
print(docs[0].page_content)
