import os
from dotenv import load_dotenv

from langchain_ollama import OllamaLLM as Ollama
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import warnings
import logging

# Suppress Deprecation Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Specifically target the transformers logger if needed
logging.getLogger("transformers").setLevel(logging.ERROR)

load_dotenv()

## Langsmith Tracking
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"    
os.environ["LANGSMITH_API_KEY"] = "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"
os.environ["LANGSMITH_PROJECT"] = "Test"

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please respond to the question asked"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title("Langchain Demo With Gemma Model")
input_text=st.text_input("What question you have in mind?")


## Ollama Llama2 model
llm=Ollama(model="gemma:2b")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))


