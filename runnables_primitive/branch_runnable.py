

from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda , RunnablePassthrough, RunnableParallel, RunnableBranch


prompt = PromptTemplate(
    template="Write a detailed report on  {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template=" Summarize the following text \n {text}",
    input_variables=['text']
)

model = ChatOllama(
    model='gemma2:2b',
    temperature=0.7
)

parser = StrOutputParser()

# report_gen_chain = RunnableSequence(prompt, model, parser)

report_gen_chain = prompt | model | parser


branch_chain = RunnableBranch(
    # (lambda x: len(x.split())>500, RunnableSequence(prompt2, model, parser)),
    (lambda x: len(x.split())>500, prompt2 | model|parser),
    RunnablePassthrough()
)


# final_chain = RunnableSequence(report_gen_chain, branch_chain)
final_chain = report_gen_chain | branch_chain

res = final_chain.invoke({"topic":"Russia VS Ukrain"})

print(res)

