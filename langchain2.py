import os

from langchain import HuggingFaceHub
from langchain.llms import Cohere
from langchain.chat_models import ChatCohere

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.schema import BaseOutputParser

from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.chains import SequentialChain

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate


os.environ["COHERE_API_KEY"]="use_your_own_api_key"


llm=Cohere(cohere_api_key=os.environ["COHERE_API_KEY"], temperature=0.7)

text="What is the capital of Kenya?"

print(llm.predict(text))

os.environ["HUGGINGFACEHUB_API_TOKEN"]="use_your_own_hf_api_key"


llm_huggingface=HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":0, "max_length":64})

output=llm_huggingface.predict("Can you tell me the capital of Germany")

print(output)

output=llm_huggingface.predict("Can you write a poem about taking a break?")
print(output)

llm.predict("Can you write a poem about taking a break?")

llm.predict("Can you tell me what the TV series 'Snowfall' is all about?")



prompt_template=PromptTemplate(
    input_variables=["country"],
    template="What is the capital of {country}?"
)

prompt_template.format(country="Kenya")


chain=LLMChain(llm=llm, prompt=prompt_template)

print(chain.run("Kenya"))

"""## Combining Multiple Chains using Simple Sequential Chain"""

capital_template=PromptTemplate(
    input_variables=["country"],
    template="What is the capital of {country}?"
)

capital_chain=LLMChain(llm=llm, prompt=capital_template)

famous_template=PromptTemplate(
    input_variables=["capital"],
    template="Who is the president of {capital}?"
)

famous_chain=LLMChain(llm=llm, prompt=famous_template)



chain=SimpleSequentialChain(chains=[capital_chain, famous_chain], verbose=True)

country="Kenya"
chain.run(country)

capital_template=PromptTemplate(
    input_variables=["country"],
    template="What is the capital of {country}?"
)

capital_chain=LLMChain(llm=llm, prompt=capital_template)

famous_template=PromptTemplate(
    input_variables=["capital"],
    template="List for me 10 amazing places to visit in {capital}?"
)

famous_chain=LLMChain(llm=llm, prompt=famous_template)



chain=SimpleSequentialChain(chains=[capital_chain, famous_chain], verbose=True)

country="Kenya"
chain.run(country)

"""## Sequential Chain"""

capital_template=PromptTemplate(
    input_variables=["country"],
    template="Please tell me the population of the capital of {country}"
)

capital_chain=LLMChain(llm=llm, prompt=capital_template, output_key="capital")

famous_template=PromptTemplate(
    input_variables=['capital'],
    template="Give me a list of 5 amazing places to visit in {country}"
)

famous_chain=LLMChain(llm=llm, prompt=famous_template, output_key="famous_places")



chain=SequentialChain(
    chains=[capital_chain, famous_chain],
    input_variables=["country"],
    output_variables=["capital", "famous_places", "country"],
    verbose=True
)

chain({"country":"Kenya"})

chain("country: Kenya")


chatllm=ChatCohere(cohere_api_key=os.environ["COHERE_API_KEY"], temperature=0.7, model='CohereForAI/c4ai-command-r-plus')


# Remove the token_count reference as it's causing the error

def _get_generation_info(self, response):
    return {
        "finish_reason": response.finish_reason,
        "generations": response.generations,
        "prompt": response.prompt,
        "search_results": getattr(response, 'search_results', None), # Handle potential absence of search_results
        "search_queries": getattr(response, 'search_queries', None), # Handle potential absence of search_queries
    }

ChatCohere.get_generation_info = _get_generation_info




try:
  result = chatllm([
    SystemMessage(content="You are an AI Comedian Assistant"),
    HumanMessage(content="Tell me some jokes on AI and relationships")
  ])
  print(result)
except Exception as e:
  print(f"An error has occurred: {e}")

"""## Prompt Template + LLM + Output Parsers"""



class Commaseperatedoutput(BaseOutputParser):
    def parse(self,text:str):
        return text.strip().split(",")


template="Your are a helpful assistant. When the use given any input , you should generate 5 words synonyms in a comma seperated list"
human_template="{text}"
chatprompt=ChatPromptTemplate.from_messages([
    ("system",template),
    ("human",human_template)
])

chain=chatprompt|chatllm|Commaseperatedoutput()

try:
  chain.invoke({"text":"intelligent"})
except Exception as e:
  print(f"Error occured: {e}")

