import os
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


# 定义函数工具，代理调用外部
@tool
def search(query: str):
    """模拟工具使用"""
    if "天气" in query.lower():
        return "今天深圳的天气为30度"
    return "今天深圳的天气为25度"

# 工具列表
tools = [search]

# 创建工具节点
tool_node = ToolNode(tools)

# 1. 初始化模型和工具，定义并绑定工具到模型
model = ChatOpenAI(base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=os.getenv("GLM_API_KEY"),
                   model="glm-4.7", temperature=0).bind_tools(tools)

# 定义函数，决定是否执行
def should_continue(state: MessagesState) -> Literal["tools", END]:
    message = state['messages']
    last_message = message[-1]
    # 若llm调用了工具，跳转到tools节点
    if last_message.tool_calls:
        return "tools"
    return END

# 调用llm
def call_model(state: MessagesState):
    message = state['messages']
    response = model.invoke(message)
    return {"messages": [response]}

# 2. 初始化状态图
builder = StateGraph(MessagesState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.set_entry_point("agent")

# 添加条件边
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

# 初始化内存，持久化图
# MemorySaver是其中一种实现
checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)

while True:
    user_input = input("User input:")
    if "exit" in user_input:
        # human-in-the-loop，消息调用传入None即可
        result = input("是否退出？(y/n)")
        if result == "y":
            break
        for event in graph.stream(None, config={"configurable": {"thread_id": 1}}, stream_mode="values"):
            event["messages"][-1].pretty_print()

    for event in graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": 1}},
        stream_mode="values"):
        event["messages"][-1].pretty_print()

# response = graph.invoke({"messages": [HumanMessage(content="今天深圳天气如何")]}, config={"configurable": {"thread_id": 1}})
# print(f"response: {response["messages"][-1].content}")
#
# response = graph.invoke({"messages": [HumanMessage(content="今天应该穿什么衣服")]}, config={"configurable": {"thread_id": 1}})
# print(f"response: {response["messages"][-1].content}")
# 生成图
graph_png = graph.get_graph().draw_mermaid_png()
with open("langgraph_hello.png", "wb") as fh:
    fh.write(graph_png)
