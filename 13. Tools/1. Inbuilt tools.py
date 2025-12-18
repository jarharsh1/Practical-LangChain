## Build in Tool Duckduckgo Search 

from langchain_community.tools import DuckDuckGoSearchRun
import warnings as warn
warn.filterwarnings('ignore')

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke(" Latest IPL Auction news.")

# print(results)


## Built in Tool - Shell Tool

from langchain_community.tools import ShellTool

shell_Tool = ShellTool()

results_shell = shell_Tool.invoke("ls")

# print(results_shell)

#################################################################################
