
from langchain_core.prompts import PromptTemplate
from langchain_classic.output_parsers import StructuredOutputParser , ResponseSchema
from langchain_ollama import ChatOllama


model = ChatOllama(
    model="gemma2:2b",
    temperature=0.7,
)


schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
    ResponseSchema(name='fact_3', description='Fact 3 about the topic'),

]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({"topic":"Black Hole"})

# print(prompt)
# res = model.invoke(prompt)

# res = parser.parse(res.content)

# print(res)

chain = template | model | parser

res = chain.invoke({"topic":"Black Hole"})

print(res)

