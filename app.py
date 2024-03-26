# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# req libs
import os
import openai
import requests
from pprint import pprint
import json
from pathlib import Path
from langchain_community.document_loaders import TextLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

# read dog site
def fetch_all_dogs():
    api_url_base = 'https://api.api-ninjas.com/v1/dogs?min_weight=1'
    headers = {'X-Api-Key': 'nHhE4rhoadk7re2SVO5sMA==Ngb2EjyF0mZoRlcj'}
    all_dogs = []
    offset = 0
    while True:
        api_url = f"{api_url_base}&offset={offset}"
        response = requests.get(api_url, headers=headers)
        if response.status_code == requests.codes.ok:
            dogs = response.json()
            if not dogs:  # Break if no more dogs are returned
                break
            all_dogs.extend(dogs)
            offset += 20  # Assuming 20 results per page
        else:
            print("Error:", response.status_code, response.text)
            break

    return all_dogs

# Fetch all dog data
dogs_data = fetch_all_dogs()
# pprint(dogs_data[1])

# Specify the output file path
output_file_path = 'dogs_data.json'

# Write the data to the JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(dogs_data, json_file, indent=4)

# This loads the JSON file into the langchain document as a text stream
loader = TextLoader(output_file_path)
documents = loader.load()

#Splitting single document into smaller pieces using RecursiveCharacterTextSplitter to leverage it with the retrieval chain
def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 300,
    length_function = tiktoken_len,
)

documents = text_splitter.split_documents(documents)
#len(documents)

#Convert text into vectors Using OpenAI's text-embedding-ada-002 for this
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)
# opted against text-embedding-3-small

openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")

# FAISS : Store the documents along with their their embeddings.
vector_store = FAISS.from_documents(documents, embeddings)

#retreiver
retriever = vector_store.as_retriever()

#custom prompt
system_template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':
If the value for barking is less than 3, then consider that dog to have low barking
If good with children value is greater than 3, then consider that good with children
Context:
{context}

Question:
{question}
"""
prompt = ChatPromptTemplate.from_template(system_template)

primary_qa_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# set up chain 
retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": prompt | primary_qa_llm, "context": itemgetter("context")}
)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
# from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools


# ChatOpenAI Templates
#system_template = """You are a helpful assistant who always speaks in a pleasant tone!
#"""

user_template = """{input}
If you don't know say you don't know.
"""


@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)

###
question = "good_with_children and name?"
result = retrieval_augmented_qa_chain.invoke({"question" : question})

print(result["response"].content)
print(result["context"])
##3
@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def main(message: cl.Message):
    settings = cl.user_session.get("settings")

    client = AsyncOpenAI()

     
    result = retrieval_augmented_qa_chain.invoke({"question" : message.content})
    msg = cl.Message(content="")

  
    # Update the prompt object with the completion
   #prompt.completion = msg.content
   #msg.prompt = prompt
    #msg.content = result


    # Send and close the message stream
    #await msg.send()
    print(result["response"].content)
    name = result["response"].content
    await cl.Message(content=f'{name}').send() 