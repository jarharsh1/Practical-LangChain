from langchain_core.tools import tool

## Creating custom tools

@tool
def add(a:int, b:int) -> int:
    """ Add two numbers"""
    return a+b

@tool
def multiply(a:int, b:int) -> int:
    """ multiply two numbers"""
    return a*b

class ToolKit:
    def get_tools(self):
        return [add,multiply]
    
toolkit = ToolKit()
tools =toolkit.get_tools()

for tool in tools:
    print(tool.name, "-->", tool.description)