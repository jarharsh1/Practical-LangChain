from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv() # Load enviornment variables from .env file

model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

result =model.invoke("Suggest me five best spy movies of all time? Just give me one reason stating why they are the best.")

print(result.content)