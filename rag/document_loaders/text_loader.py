from langchain_community.document_loaders import TextLoader
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
loader = TextLoader('poem_cricket.txt', encoding='utf-8')

docs = loader.load()

print(type(docs))
print(len(docs))

print(type(docs[0]))
# print(docs[0])
print(docs[0].page_content)

chain = prompt | model | parser 

res = chain.invoke({"poem":docs[0].page_content})

print(res)