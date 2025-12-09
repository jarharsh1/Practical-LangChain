from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "google/gemma-2-2b-it",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age, gender, email id of an imaginary person \n {format_instructions}",
    input_variables = [],
    partial_variables = {'format_instructions':parser.get_format_instructions()}
)


prompt = template.format()

output = model.invoke(prompt)

final_output = parser.parse(output.content)

print("Generating using only JsonSchema method only \n",final_output)

chain = template | model | parser

result = chain.invoke({})

print("Generating using only JsonSchema method using chains \n",result)

template1 = PromptTemplate(
    template="Give me random facts about {topic} \n {format_instructions}",
    input_variables = ['topic'],
    partial_variables = {'format_instructions':parser.get_format_instructions()}
)

chain = template1 | model | parser

result = chain.invoke({'Bhagalpur'})

print("Generating without any restriction on the output format \n",result)



