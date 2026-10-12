from langchain_groq import ChatGroq

from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model='openai/gpt-oss-20b', temperature=0.7)

result = llm.invoke("Suggest me 5 indian male names")

print(result.content)

