from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

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
                template="write a 5 line summary on the following text {text}",
                input_variables=['text']
            )

prompt1 = template1.invoke({'topic':'TVS Apache RTR 160'})

model_response1 = model.invoke(prompt1)

prompt2 = template2.invoke({"text":model_response1.content})

model_response2 = model.invoke(prompt2)

print(model_response1.content)
print("--------------------------------------------------")

print(model_response2.content)