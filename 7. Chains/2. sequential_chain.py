from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv() ## Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task = "text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template ="Generate me detailed report of {description} ",
    input_variables = ['description'],
)

prompt2 = PromptTemplate(
    template ="Generate me a 5 pointer summary for the following report:\n{report}",
    input_variables = ['report'],
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'description':"Is there an Islamisation of Bollywood?"})

print(result)

chain.get_graph().print_ascii()