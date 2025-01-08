import os
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
import json

# with open('config.json', 'r') as file:
#     config = json.load(file)
#     togetherAI_api_key = config["togetherAI_api_key"]
#     OPEN_AI_API_KEY = config["OpenAI_api_key"]


#os.environ['TOGETHER_API_KEY'] = togetherAI_api_key


AVALAI_BASE_URL = "https://api.avalai.ir/v1"
GPT_MODEL_NAME = "gpt-4o-mini"
model = ChatOpenAI(model=GPT_MODEL_NAME,
                        base_url=AVALAI_BASE_URL,
                        api_key=os.environ["AVALAI_API_KEY"])

# MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
# model = ChatOpenAI(
#   api_key=os.environ["TOGETHER_API_KEY"],
#   base_url="https://api.together.xyz/v1",
#   model=MODEL_NAME
# )



def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    print("-----------MULTIPLY Tool----------")
    print("multiplying {} & {}".format(a, b))
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b



def search_tool(query: str):
    """Search the web for the query.
    Args:
        query: The query to search for.
    """
    print("-----------SEARCHING Tool----------")
    print("searching for", query)
    results = TavilySearchResults(max_results=3).invoke(query)
    print("results", results)
    return results



tools = [add, multiply, divide, search_tool]
llm_with_tools = model.bind_tools(tools)



sys_msg = SystemMessage(
    content="You are a helpful assistant tasked with using search or performing arithmetic on a set of inputs.")

def reasoner(state: MessagesState):
    print("-------------Reasoner----------------")
    messages = [sys_msg] + state["messages"]
    response = llm_with_tools.invoke(messages)
    print("Response: ")
    print(response)
    return {"messages": [response]}




# Graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("reasoner", reasoner)
builder.add_node("tools", ToolNode(tools))

# Add edges
builder.add_edge(START, "reasoner")
builder.add_conditional_edges(
    "reasoner",
    # If the latest message (result) from node reasoner is a tool call -> tools_condition routes to tools
    # If the latest message (result) from node reasoner is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "reasoner")
react_graph = builder.compile()


# messages = [HumanMessage(content="What is 2 times Milad Mohammadi's age?")]
# messages = react_graph.invoke({"messages": messages})
