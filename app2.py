## Conversational Q&A Chatbot
import streamlit as st
import os

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chat_models import ChatCohere

## Streamlit UI
st.set_page_config(page_title="Conversational Q&A Chatbot")
st.header("Hey, Let's Chat")

from dotenv import load_dotenv
load_dotenv()

COHERE_API_KEY="zhMrOjPTMRsBlwEcx58f6rkEluu3WUiHetNW2xEX"
HUGGINGFACE_HUB_API_TOKEN="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"


chat=ChatCohere(temperature=0.7)

if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages']=[
        SystemMessage(content="Yor are a comedian AI assitant"),
        HumanMessage(content="Tell me a joke about AI and relationships")
    ]

## Function to load OpenAI model and get respones

def get_chatmodel_response(question):
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    answer=chat(st.session_state['flowmessages'])
    st.session_state['flowmessages'].append(AIMessage(content=answer.content))
    return answer.content

input=st.text_input("Input: ",key="input")
response=get_chatmodel_response(input)

submit=st.button("Ask the question")

## If ask button is clicked

if submit:
    st.subheader("The Response is")
    st.write(response)