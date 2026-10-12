from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser


model = ChatOllama(
    model = 'gemma2:2b',
    temperature=0.7,
)

prompt = PromptTemplate(
    template="Generate a detailed report about {topic}",
    input_variables=['topic']
)

prompt1 = PromptTemplate(
    template="Generate the 5 line summary about {text}",
    input_variables=['text']
)

parser = StrOutputParser()

chain = prompt | model | parser | prompt1 | model | parser


res = chain.invoke({"topic":"Habits"})

print(res)


chain.get_graph().print_ascii()    # it visuals the chain 
