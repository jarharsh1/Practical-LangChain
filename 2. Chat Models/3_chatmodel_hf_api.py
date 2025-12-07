from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv() # Load environment variables

# 1. Define the LLM (Fixed typo: HuggingFaceEndpoint)
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # Changed to a more reliable model for Chat
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

# 2. Wrap it in the Chat interface
model = ChatHuggingFace(llm=llm)

# 3. Invoke
result = model.invoke("Suggest me five best spy movies of all time? Just give me one reason stating why they are the best.")

print(result.content)