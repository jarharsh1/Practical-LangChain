from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# 1. Setup Model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name : str = Field(description='Name of the person')
    age :int = Field(gt=18,description='age of the person')
    salary: float = Field(gt=100000, lt=400000, description='Salary of the person')
    state : str = Field(description='State of the person')
    pin : str = Field(
        pattern=r'^8\d{5}$', 
        description='Pincode of the person (must start with 8 and be 6 digits)'
    )

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template = "Give me the name, age, salary, state and pin code of an imaginary person from {country} \n {format_instructions}",
    input_variables = ['country'],
    partial_variables = {'format_instructions':parser.get_format_instructions()}
    
)

prompt = template.invoke({'country':'India'})

print("Prompt:\n",prompt)

model_output = model.invoke(prompt)

final_output = parser.parse(model_output.content)

print(final_output)
print("~~~~~~~~~~~")

chain = template | model | parser

result = chain.invoke({'country':"Gotham"})

print(result)