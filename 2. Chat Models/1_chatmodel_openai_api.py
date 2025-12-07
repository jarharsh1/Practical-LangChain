from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv() # Load enviornment variables from .env file

model = ChatOpenAI(model="gpt-4",
                   temperature=0.5,
                   max_completion_tokens=10)

result = model.invoke("Suggest me five best Biscuit brands in India?")

print(result.content)