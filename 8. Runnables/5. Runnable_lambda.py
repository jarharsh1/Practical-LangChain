from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableLambda

load_dotenv()  ## Load environment variables from .env file

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template= "Create a joke from this topic:\n{joke_topic}",
    input_variables=['joke_topic'],
)

parser = StrOutputParser()

def word_counter(text:str) ->int:
    
    return len(text.split())

word_counter_runnable = RunnableLambda(word_counter)

joke_chain = RunnableSequence(prompt, model, parser)

full_chain = RunnableParallel({
    "joke": RunnableSequence(joke_chain,RunnablePassthrough()),
    "joke_word_count": RunnableSequence(joke_chain, word_counter_runnable)
}
)

result = full_chain.invoke({'joke_topic': 'Cricket'})

print("Generated Joke and Word Count:\n", result)