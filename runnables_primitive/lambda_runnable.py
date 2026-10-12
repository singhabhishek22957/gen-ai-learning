# from langchain_core.runnables import RunnableLambda

def word_counter(text):
    return len(text.split())

# runnable_word_count = RunnableLambda(word_counter)

# res = runnable_word_count.invoke("Hi, I am abhishek")

# print(res)  # output 4


from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda , RunnablePassthrough, RunnableParallel


prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=['topic']
)




model = ChatOllama(
    model='gemma2:2b',
    temperature=0.7
)

parser = StrOutputParser()


joke_gen_chain = RunnableSequence(prompt, model, parser)


parallel_chain = RunnableParallel({
    "joke":RunnablePassthrough(),
    "word_count":RunnableLambda(word_counter)
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

res = final_chain.invoke({"topic":"AI"})

final_res = """ {} \nword count - {}""".format(res['joke'], res['word_count'])
print(final_res)