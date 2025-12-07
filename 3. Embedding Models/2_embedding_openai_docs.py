from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv() ## calling reference files & keys from the environment

documents =['Delhi is the capital of India', 'The capital of USA is Washington DC', 'The capital of UK is London']

embedding = OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)


## embed_query for single query

result = embedding.embed_documents(documents)

print(str(result))