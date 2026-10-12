from langchain_ollama import ChatOllama

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel , Field 
from typing import Literal
from langchain_core.runnables import RunnableParallel , RunnableBranch, RunnableLambda

class Feedback(BaseModel):
    sentiment: Literal['positive', "negative"]  = Field(description="Give the sentiment of the feedback")


model = ChatOllama(
    model="gemma3:4b",
    temperature=0,
)

parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text positive or negative \n {feedback}   \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={"format_instruction":parser2.get_format_instructions()}
)


prompt2 = PromptTemplate(
    template=" Write a n appropriate response to this positive feedback \n {feedback} ",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template=" Write a n appropriate response to this negative feedback \n {feedback} ",
    input_variables=['feedback']
)



# print(prompt1)
classifier_chain = prompt1 | model | parser2

branch_chain = RunnableBranch(
    ( lambda x:x.sentiment=='positive', prompt2 | model | parser),
    ( lambda x:x.sentiment=='negative', prompt3 | model | parser),
    RunnableLambda(lambda X: "Could not find sentiment")
)


chain = classifier_chain | branch_chain 

res = chain.invoke({"feedback":"This is terrible phone "})

print(res)

chain.get_graph().print_ascii()    

