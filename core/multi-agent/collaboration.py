"""
多Agent协同工作，实现：搜索进十年来AI市场规模，并最终生成图表
1. Search Agent
2. Chart Generation Agent
3. call_tool
4. Router分发
"""
import functools
import operator
import os
from typing import TypedDict, Annotated, Sequence, Literal

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode


# 状态对象，在图的每个节点之间传递
class AgentState(TypedDict):
    # messages存储消息序列，并通过operator.add实现累加
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # sender 用于存储当前消息的发送者，指导当前消息是由哪个代理生成的
    sender: str

def agent_node(state: AgentState, agent, name: str) -> AgentState:
    # 调用代理
    result = agent.invoke(state)
    # 若非ToolMessage类型（即Tavily搜索出来的结果，需要大模型处理数据）
    #   需将tavily result 转换为AIMessage类型，并将name作为消息发送者名称附加到上下文中
    if not isinstance(result, ToolMessage):
        result = AIMessage(**result.model_dump(exclude={"type", "name"}), name=name)
    return {
        "messages": [result],
        # 可以通过sender跟踪发送者，以指导下一个消息接受者
        "sender": name,
    }

glm_model = ChatOpenAI(base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=os.getenv("GLM_API_KEY"),
                   model="glm-4.7", temperature=0)
minimax_model = ChatOpenAI(base_url="https://api.minimaxi.com/v1", api_key=os.getenv("MINIMAX_API_KEY"),
                   model="MiniMax-M2.7", temperature=0.1)

tavily_tool = TavilySearch(max_results=5)
tools = [tavily_tool]
tool_node = ToolNode(tools)

search_prompt = """
你是一个擅长搜集、总结AI行业热点信息的助手，且善于与其他助手合作
使用提供的工具来推进问题的回答。
如果你认为你提供的回答置信度不高，或难以回答，总结当前收集的信息，后续会有其他拥有不同工具的助手基于你的总结继续执行。
若你或其他助手，取得置信度大于70%的答案，在你的回答前加上FINAL_ANSWER，以便团队指导任务停止
"""

draw_chart_prompt = """
你是一个擅长根据历史数据生成数据图表的AI助手，
使用已有的数据根据要求进行绘图，
若条件不满足，可以继续调用工具搜索
"""

# 创建检索节点和图表生成节点
research_agent = create_agent(glm_model, tools, system_prompt=search_prompt)
draw_chart_agent = create_agent(minimax_model, tools, system_prompt=draw_chart_prompt)
research_node = functools.partial(agent_node, agent=research_agent, name="researcher")
draw_chart_node = functools.partial(agent_node, agent=draw_chart_agent, name="draw_chart")

def router(state: AgentState) -> Literal["call_tool", "__end__", "continue"]:
    messages = state["messages"]
    last_message = messages[-1]
    # 检查最后一条消息是否包含工具调用
    if last_message.tool_calls:
        return "call_tool"
    if "FINAL_ANSWER" in last_message.content:
        return "__end__"
    return "continue"

builder = StateGraph(AgentState)
builder.add_node("researcher", research_node)
builder.add_node("draw_chart", draw_chart_node)
builder.add_node("call_tool", tool_node)
builder.set_entry_point("researcher")

builder.add_conditional_edges(
    "researcher",
    router,
    {"continue": "draw_chart", "call_tool": "call_tool", "__end__": END},
)

builder.add_conditional_edges(
    "draw_chart",
    router,
    {"continue": "researcher", "call_tool": "call_tool", "__end__": END},
)

builder.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
    {"researcher": "researcher", "draw_chart": "draw_chart"}
)

graph = builder.compile()

# graph_png = graph.get_graph().draw_mermaid_png()
# with open("collaboration.png", "wb") as fh:
#     fh.write(graph_png)

events = graph.stream(
    {
        "messages": [
            HumanMessage(
                content="获取过去5年AI软件市场规模，然后绘制一条折线图，若绘制完成，则任务完成"
            )
        ]
    },
    {"recursion_limit": 150},
)

for s in events:
    print(s)
    print("==============")
