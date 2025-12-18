## custom tools

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required =True, description = 'First number to be added')
    b: int = Field(required =True, description = 'Second number to be added')
    
def multiply_func(a:int, b:int) -> int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func = multiply_func,
    name = "multiply",
    description = "Multiply two numbers",
    args_schema = MultiplyInput
)

result = multiply_tool.invoke({'a':3,'b':5})

print(result)
print(f"Tools name: {multiply_tool.name}")
print(f"Tools description: {multiply_tool.description}")
print(f"Tools argument: {multiply_tool.args}")