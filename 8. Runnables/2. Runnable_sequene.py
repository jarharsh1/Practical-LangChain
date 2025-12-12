from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()  ## Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    )

model = ChatHuggingFace(llm=llm)

input_prompt = PromptTemplate(
    template = "Write a joke about the following subject:\n{joke_subject}",
    input_variables=['joke_subject'],   
)

summary_prompt = PromptTemplate(
    template = "Explain me the joke in detailed way: \n{joke}",
    input_variables=['joke']
)

parser = StrOutputParser()

chain = RunnableSequence(input_prompt, model,  parser, summary_prompt, model , parser)

result = chain.invoke({'joke_subject': 'cats'})

print(result)