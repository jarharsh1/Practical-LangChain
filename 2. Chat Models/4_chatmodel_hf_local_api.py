from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

import os

os.environ["HF_home"] = "C:/Users/harsh.raj/OneDrive - Aster DM Healthcare/Codes/LangChain/Repository"

llm = HuggingFacePipeline.from_model_id(
    model_id = "HuggingFaceH4/zephyr-7b-beta",
    task ="text-generation",
    pipeline_kwargs={"temperature":0.7,
                     "max_new_tokens":512}
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("Suggest me five best spy movies of all time? Just give me one reason stating why they are the best.")

print(result.content)