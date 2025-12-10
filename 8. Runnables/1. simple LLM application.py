from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()  ## Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="Generate a creative story based on the following prompt:\n{story_prompt}",
    input_variables=['story_prompt'],
)

input = input("Enter a story prompt:   ")

chain = prompt | model

result = chain.invoke({'story_prompt': input})

print("Generated Story:\n", result)
