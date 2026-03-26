import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

glm_model = ChatOpenAI(base_url="https://open.bigmodel.cn/api/paas/v4/", api_key=os.getenv("GLM_API_KEY"),
                   model="glm-4.7", temperature=0)
minimax_model = ChatOpenAI(base_url="https://api.minimaxi.com/v1", api_key=os.getenv("MINIMAX_API_KEY"),
                   model="MiniMax-M2.7", temperature=0.1)

tools = []

prompt = """
你是一个删除搜集、总结AI行业热点信息的助手，且善于与其他助手合作
使用提供的工具来推进问题的回答。
如果你认为你提供的回答置信度不高，或难以回答，总结当前收集的信息，后续会有其他助手基于你的总结继续执行。
若你或其他助手，取得置信度大于70%的答案，在你的回答前加上FINAL_ANSWER，以便团队指导任务停止
"""

create_agent(glm_model, tools, system_prompt=prompt)
