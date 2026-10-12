
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama


model = ChatOllama(
    model="qwen3:0.6b",
    temperature=0.7,
)


parser = JsonOutputParser()

template = PromptTemplate(
    template="give me the name, age  and city of a fictional person /n  {format_instruction}",
    input_variables=[],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

template1 = PromptTemplate(
    template="give me 5 facts about {topic} /n  {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

# prompt = template.format()

# res = model.invoke(prompt)

# res = parser.parse(res.content)

# print(type(res))
# print(res)
# print(res['name'])


chain = template | model | parser 

# res = chain.invoke({})     # if you have not input field then you need to send empty dictionary 

# print(type(res))
# print(res)
# print(res['name'])


chain1 = template1 | model |parser 

res = chain1.invoke({"topic": "Black Hole"})


print(type(res))
print(res)








