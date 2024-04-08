from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI()

prompt = ChatPromptTemplate.from_messages([
  ("system", "You are world class technical documentation writer."),
  ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

response = chain.invoke({ "input": "do I have to test my application before deploy?" })

print(response)
