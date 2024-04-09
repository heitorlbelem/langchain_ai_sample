from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI()

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
embeddings = OpenAIEmbeddings()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()
chat_history = [
  HumanMessage(content="Can LangSmith help test my LLM applications?"),
  AIMessage(content="Yes!")
]

prompt = ChatPromptTemplate.from_messages([
  ("system", "Answer the user's questions based on the below context:\n\n{context}"),
  MessagesPlaceholder(variable_name="chat_history"),
  ("user", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({
  "chat_history": chat_history,
  "input": "Give me step by step",
  "context": retriever_chain
})

print(response['answer'])
