from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Explain the following joke {text}",
    input_variables=['text']
)

model = ChatOllama(
    model='gemma2:2b',
    temperature=0.7
)

parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser , prompt2, model , parser)

res = chain.invoke({"topic":"AI"})
print(res)