from langchain_huggingface import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv() ## calling reference files & keys from the environment

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction")

texts =['Delhi is the capital of India', 'The capital of USA is Washington DC', 'The capital of UK is London']

result = embedding.embed_documents(texts)

print(str(result))