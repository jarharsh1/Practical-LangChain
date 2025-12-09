from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

# --- CORRECT IMPORT (Requires 'pip install langchain') ---
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

load_dotenv()

# 1. Setup Model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

# 2. Define Schema
schema = [
    ResponseSchema(name='fact_1', description="Fact_1 about the topic"),
    ResponseSchema(name='fact_2', description="Fact_2 about the topic"),
    ResponseSchema(name='fact_3', description="Fact_3 about the topic")
]

# 3. Initialize Parser
parser = StructuredOutputParser.from_response_schemas(schema)

# 4. Define Template
template = PromptTemplate(
    template="Give me random facts about {topic}\n{format_instructions}",
    input_variables=['topic'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# 5. Execution
prompt = template.invoke({'topic': 'Bhagalpur'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)