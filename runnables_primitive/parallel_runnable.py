from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel



prompt1 = PromptTemplate(
    template="Generate a tweet about {topic} ",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate linkdin post{topic}",
    input_variables=['topic']
)

model = ChatOllama(
    model='gemma2:2b',
    temperature=0.7
)

parser = StrOutputParser()


parallel_chain = RunnableParallel({
    "tweet": RunnableSequence(prompt1, model, parser),
    "linkdin": RunnableSequence( prompt2 , model, parser)
})


res = parallel_chain.invoke({"topic":"AI"})

print( res)