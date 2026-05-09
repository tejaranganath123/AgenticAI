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

## Langsmith Tracking - get these  while running the code to track the interactions with the model in langsmith dashboard


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


