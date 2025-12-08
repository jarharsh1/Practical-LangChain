from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field

load_dotenv()  # Load environment variables from .env file

class Review(TypedDict):
    bike_name:str
    bike_brand:str
    summary:str
    sentiment:str
 
 
json_schema={
  "title": "Review_json_schema",
  "type": "object",
  "properties": {
    "productName": {
      "type": "string",
      "description": "Derived name of the product being reviewed"
    },
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes mentioned in the review"
    },
    "summary": {
      "type": "string",
      "description": "Summary of the review made about the product"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg", "neu"],
      "description": "Overall user sentiment, Positive, negative or neutral"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Any pros mentioned about the product in the review"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Any cons mentioned about the product in the review"
    }
  },
  "required": ["productName", "key_themes", "summary", "sentiment","cons"]
}

    
    
model = ChatOpenAI()
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""Best in handling. Good style and design. Moderate in comfort. ( Can create back pain if ride long). 
                         Since is has 150cc so you can't go rough above 70km/hr or else bear with the low mileage. To maintain better be at 50-60km/hr.
                         My got 35km/litre. Every part has a replacement date, Oil is must to be changed on a fixed interval or can lead to engine break.
                         But still have a good experience with the pulsar since it is a strong and powerful bike.""")

# print("\n Typed Dict \n")
# print(result)
print("~~~~~~~~~~~")

structured_model_annotated = model.with_structured_output(json_schema,method = "json_mode")

result = structured_model_annotated.invoke("""Switched s25 ultra after getting a good deal. my last samsung was s10. Its an amazing phone. Battery life is very good (not great, i have had find x8 pro), cameras are very dependable and more natural, i see things have changed with samsung, it used to be over saturated (not ultra processed like x8 pro or xiaomi 15 with leica vivid which i have had in the past). what i liked most is the software, oneui have come a long way its very polish and lot of ai stuff are very usable. Also samsung software is really good they are not bloatware anymore. for example the browser, files and note app are way better than google ones. Lot of attention to detail compared to chinese software. s-pen is handy, but im not sure if i will be using it much. """)


print("\njson schema \n")
print(result)
print("~~~~~~~~~~~")
