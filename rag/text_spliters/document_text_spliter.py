from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('pdf.pdf')
docs = loader.load()

text = """

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel , Field


model = ChatOllama(
    model="gemma2:2b",
    temperature=0.7,
)

class Person(BaseModel):
    name:str = Field(description="Name of the person")
    age: int = Field(gt=10, description="Age of the person")
    city:str = Field(description="Name of the city the person belongs to ")



parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name , age , city of a fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({"place":'Indian'})
# print("prompt", prompt)
# res = model.invoke(prompt)

# res = parser.parse(res.content)

# print(res)



chain = template | model |parser

res = chain.invoke({'place':"American"})
print(res)


"""


splitters = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 300,
    chunk_overlap=0,
)



# res = splitters.split_text(text)
res = splitters.split_text(text)

for r in res :
    print(r, end='\n')