import chainlit as cl
from ReAct_graph import builder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.schema.runnable.config import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver


memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


@cl.on_chat_start
async def on_chat_start():
    model = graph
    cl.user_session.set("runnable", model)
    cl.Message(content="Hello! I am a chatbot. How can I help you?").send()

@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()

    # graph.get_state(config)
    final_answer = cl.Message(content="")
    this_time_messages = []    
    for msg, metadata in graph.stream({"messages": HumanMessage(content=message.content)}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        this_time_messages.append(msg)
        if not isinstance(msg, AIMessage):
            continue
        await final_answer.stream_token(msg.content)
    cl.user_session.set("messages", this_time_messages)
    await final_answer.send()