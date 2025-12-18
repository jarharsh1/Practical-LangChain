## custom tools

from langchain_core.tools import BaseTool
from pydantic import BaseModel,Field
from typing import Type

class MultiplyInput(BaseModel):
    a: int = Field(required =True, description = 'First number to be added')
    b: int = Field(required =True, description = 'Second number to be added')
    
class MultiplyTool(BaseTool):
    name:str = "multiply"
    description:str = "Multiply two numbers"
    
    args_schema :Type[BaseModel] = MultiplyInput
    
    def _run(self,a:int,b:int)-> int:
        return a*b
    
multiply_tool = MultiplyTool() 

result = multiply_tool.invoke({'a':3,'b':5})

print(result)
print(f"Tools name: {multiply_tool.name}")
print(f"Tools description: {multiply_tool.description}")
print(f"Tools argument: {multiply_tool.args}")