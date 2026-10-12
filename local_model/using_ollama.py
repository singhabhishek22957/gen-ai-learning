
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOllama(
    model="qwen3:0.6b",
    temperature=0.7,
)

messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="Explain LangChain in simple words.")
]

response = llm.invoke(messages)

print(response.content)
