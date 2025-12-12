from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel

load_dotenv()  ## Load environment variables from .env file

# Model 1: Llama 3.1 – used for notes
llm1 = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",  # IMPORTANT: no :cerebras
    task="conversational",                       # provider only supports conversational
    max_new_tokens=256,
    do_sample=False,
)

model1 = ChatHuggingFace(llm=llm1)

# Model 2: Mistral 7B – used for questions
llm2 = HuggingFaceEndpoint(
    repo_id="EssentialAI/rnj-1-instruct",
    task="conversational",                       # also used via chat API
    max_new_tokens=256,
    do_sample=False,
)

model2 = ChatHuggingFace(llm=llm2)

# ------------- Prompts -------------

prompt1 = PromptTemplate(
    template="Generate a tweet on following topic:\n\n{topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Generate a detailed LinkedIn post based on the following topic:\n\n{topic}",
    input_variables=["tweet"],
)

parser = StrOutputParser()
parser_json = JsonOutputParser()

# ------------- Chains (parallel + merge) -------------
parallel_chain = RunnableParallel({
    "tweet":RunnableSequence(prompt1, model1, parser),      # tweet branch
    "Linkedin":RunnableSequence(prompt2, model2, parser)   # LinkedIn post branch
})

result = parallel_chain.invoke({'topic': 'Artificial Intelligence in Healthcare'})

print("Generated Outputs:\n", result)
