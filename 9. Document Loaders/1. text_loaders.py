from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()  ## Load environment variables from .env file

# Optional: You can change the directory, though it is not strictly needed 
# since you are using the full absolute path below.
try:
    os.chdir("./Practical-LangChain/9. Document Loaders")
except FileNotFoundError:
    print("Directory not found, skipping directory change.")

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

# Initialize the loader with the file path
loader = TextLoader(r"C:\Users\harsh.raj\OneDrive - Aster DM Healthcare\Codes\LangChain\Practical-LangChain\9. Document Loaders\cricket.txt", encoding="utf-8")

# FIX: .load() takes no arguments
docs = loader.load()

print(docs[0].page_content)
print("\n")
print(docs[0].metadata)

chain = prompt | model | parser


result = chain.invoke({'text': docs[0].page_content})
print("Final summary:\n", result)

chain.get_graph().print_ascii()
