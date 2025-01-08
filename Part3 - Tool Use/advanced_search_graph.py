import os
from typing import TypedDict, Optional
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_community.tools.tavily_search import TavilySearchResults
# Constants for API configuration
AVALAI_BASE_URL = "https://api.avalai.ir/v1"
GPT_MODEL_NAME = "gpt-4o-mini"

# Initialize the ChatOpenAI instance
gpt4o_chat = ChatOpenAI(
    model=GPT_MODEL_NAME,
    base_url=AVALAI_BASE_URL,
    api_key=os.environ["AVALAI_API_KEY"]
)

class MyState(TypedDict):
    user_query: str
    rewritten_query: Optional[str]
    search_results: Optional[str]
    final_answer: Optional[str]




# Define node functions
def node_query_rewrite(state: MyState) -> MyState:
    """
    Rewrites the user's query using an LLM for clarity or correctness.
    """
    print("---Node Query Rewrite---")
    user_query = state["user_query"]
    
    rewrite_prompt = f"Please rewrite this query suitable for search engines:\n\n{user_query}"
    rewritten = gpt4o_chat.invoke(rewrite_prompt).content
    print("Rewritten query:", rewritten)
    state["rewritten_query"] = rewritten
    return state

def node_search_internet(state: MyState) -> MyState:
    """
    Searches the internet for the rewritten query using a search tool.
    """
    print("---Node Search Internet---")
    rewritten_query = state["rewritten_query"]
    
    # Implement your own search_tool function or import it
    results = TavilySearchResults(max_results=3).invoke(rewritten_query)
    print("Search results:")
    print(results)
    state["search_results"] = str(results)
    return state

def node_generate_answer(state: MyState) -> MyState:
    """
    Generates a final answer based on the original user query and the search results.
    """
    print("---Node Generate Answer---")
    user_query = state["user_query"]
    search_info = state["search_results"] or "No search results found."
    
    final_answer = gpt4o_chat.invoke(
        f"User's query: {user_query}\n\nSearch results:\n{search_info}\n\n"
        "Please provide a helpful answer."
    )
    print("Final answer:", final_answer.content)
    state["final_answer"] = final_answer.content
    return state

def get_advanced_search_graph() -> StateGraph:
    """
    Constructs and returns the advanced search graph.
    """
    # Build the graph
    builder = StateGraph(MyState)

    # Add nodes
    builder.add_node("node_query_rewrite", node_query_rewrite)
    builder.add_node("node_search_internet", node_search_internet)
    builder.add_node("node_generate_answer", node_generate_answer)

    # Add edges (linear flow)
    builder.add_edge(START, "node_query_rewrite")
    builder.add_edge("node_query_rewrite", "node_search_internet")
    builder.add_edge("node_search_internet", "node_generate_answer")
    builder.add_edge("node_generate_answer", END)

    # Compile and return the graph
    return builder.compile()