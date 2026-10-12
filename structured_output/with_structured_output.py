from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict , Annotated

load_dotenv()

model = ChatGroq(model='openai/gpt-oss-20b', temperature=0.7)

# schema 
# class Review(TypedDict):
#     summary:str
#     sentiment:str
class Review(TypedDict):
    summary:Annotated[str, "The summary of the review"]
    sentiment:Annotated[str, "The sentiment of the review"]
    key_themes:Annotated[list[str], "The key themes of the review"]

structured_model = model.with_structured_output(Review)


result = model.invoke("The phone has a beautiful design and an excellent battery life. The camera takes clear pictures, but performance becomes slow when multiple apps are running. Overall, it offers good value for money, although there is room for improvement.")



print (result)