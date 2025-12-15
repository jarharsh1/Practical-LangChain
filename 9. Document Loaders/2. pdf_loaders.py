from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()  ## Load environment variables from .env file

try:
    os.chdir("./Practical-LangChain/9. Document Loaders")
except FileNotFoundError:
    print("Directory not found, skipping directory change.")

loader = PyPDFLoader(r"C:\Users\harsh.raj\OneDrive - Aster DM Healthcare\Codes\LangChain\Practical-LangChain\9. Document Loaders\Panchatantra-.pdf")

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template = "Summarize the following text into points:\n{text}",
    input_variables=['text'],
)

parser = StrOutputParser()

docs = loader.load()

# print(docs)
print("Total number of pages: ",len(docs))

print(docs[1].metadata)