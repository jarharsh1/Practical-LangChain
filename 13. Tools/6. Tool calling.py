from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# -------------------------
# Tool definition
# -------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Given two numbers a and b, return their product."""
    return a * b


# -------------------------
# Bind tool to LLM
# -------------------------
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([multiply])


# -------------------------
# 1️⃣ Human message
# -------------------------
human_msg = HumanMessage(content="Multiply 2 and 6")
messages = [human_msg]


# -------------------------
# 2️⃣ AI message (tool call)
# -------------------------
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

print("AI message:")
print(ai_msg)


# -------------------------
# 3️⃣ Tool execution + ToolMessage
# -------------------------
tool_call = ai_msg.tool_calls[0]          # guaranteed tool call
tool_output = multiply.invoke(tool_call["args"])

tool_msg = ToolMessage(
    content=str(tool_output),
    tool_call_id=tool_call["id"]
)

messages.append(tool_msg)

print("\nTool message:")
print(tool_msg)


# -------------------------
# Final LLM response (uses tool output)
# -------------------------
final_response = llm_with_tools.invoke(messages)

print("\nFinal response:")
print(final_response)
