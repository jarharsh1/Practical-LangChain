## custom tools

from langchain_community.tools import tool

## Steps 1 : create a function

def multiply(a,b):
    """Multiply two numbers"""
    return a*b

## Step 2 : add type hinting 

def multiply(a:int, b:int) -> int:
    """ Multiply two numbers"""
    return a*b

## Step 3:  add tool decorator

@tool
def multiply(a:int, b:int) -> int:
    """ Multiply two numbers"""
    return a*b

result = multiply.invoke({'a':5,'b':6})

print(result)
print(f"Tools name: {multiply.name}")
print(f"Tools description: {multiply.description}")
print(f"Tools argument: {multiply.args}")
