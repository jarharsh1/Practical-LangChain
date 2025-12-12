from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda, RunnableBranch


load_dotenv()  ## Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template = "Write a detailed report on the following topic:\n{report_topic}",
    input_variables=['report_topic'],
)

prompt2 = PromptTemplate(
    template = "Create a concise summary for the following topic:\n{summary_topic}",
    input_variables=['summary_topic'],
)

parser = StrOutputParser()

initial_chain = RunnableSequence(prompt1, model, parser)

Branch_chain = RunnableBranch(
    (lambda x:len(x.split())>300, RunnableSequence(prompt2,model,parser)),
    RunnableSequence(initial_chain, RunnablePassthrough())
)

final_chain = RunnableSequence(initial_chain,Branch_chain)

result = final_chain.invoke({'report_topic': 'Tariff policies of USA for India'})

print("Final Output:\n", result)
