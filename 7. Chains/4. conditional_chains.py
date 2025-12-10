from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnableBranch, RunnableLambda


load_dotenv() ## Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['Positive','Negative'] = Field(description='Sentiment of the feedback')
    
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template ="Classify the sentiment of the following text into positive or negative \n {feedback} format the response as {format_instructions}",
    input_variables = ['feedback'],
    partial_variables = {'format_instructions':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

result = classifier_chain.invoke({'feedback':"The product quality is excellent and I am very satisfied!"}).sentiment

print("Classifier Result:\n",result)

good_prompt = PromptTemplate(
    template = "Write only 1 message for the following positive feedback:\n {feedback}",
    input_variables = ['feedback'],
)

bad_prompt = PromptTemplate(
    template = "Write only 1 message for the following negative feedback:\n {feedback}",
    input_variables = ['feedback'],
)

conditional_branch = RunnableBranch(
    (lambda x:x.sentiment=='Positive', good_prompt | model | parser),
    (lambda x:x.sentiment=='Negative', bad_prompt | model | parser),
    RunnableLambda(lambda x:'Could not find the sentiment')
)

chain = classifier_chain | conditional_branch

final_result = chain.invoke({'feedback':"This is the worst service I have ever experienced."})

print("Final Result:\n",final_result)

chain.get_graph().print_ascii()