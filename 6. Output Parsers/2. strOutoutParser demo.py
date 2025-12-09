from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv() ## loading APIs

llm = HuggingFaceEndpoint(
    repo_id = "HuggingFaceH4/zephyr-7b-beta",
    task = "text-generation"
)

model = ChatHuggingFace(llm = llm)

#1. prompt

template1 = PromptTemplate(
                template="write a detailed summary about {topic}",
                input_variables=['topic']
            )

#2. prompt

template2 = PromptTemplate(
                template="write a 5 point summary on the following text {text}",
                input_variables=['text']
            )

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'Samsung S25 Ultra'})

print(result)