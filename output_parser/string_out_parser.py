
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="qwen3:0.6b",
    temperature=0.7,
)

template = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=['topic']
)


template1 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n{text}",
    input_variables=['text']
)


# without output parser 


# prompt1 = template.invoke({"topic":"Black Hole"})

# res = model.invoke(prompt1)


# prompt2 = template1.invoke({"text":res.content})

# res1 = model.invoke(prompt2)

# print(res1.content)


#  with output parser 

parser = StrOutputParser()

chain = template | model | parser | template1 | model | parser 

res = chain.invoke({"topic":"Black Hole"})

print(res)





