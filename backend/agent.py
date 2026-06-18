"""
Agent 模块

实现基于 ReAct 模式的 AI Agent，支持 Function Calling。
"""

import json
from typing import List, Dict, Any, Optional, Callable
from openai import OpenAI

from .config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, validate_config
from .retriever import Retriever, format_context
from .generator import Generator
from pathlib import Path
from .config import BASE_DIR


# 工具定义（JSON Schema 格式）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "从知识库中检索相关文档，用于回答用户问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询文本"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回的最大结果数，默认为 5",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_doc",
            "description": "总结指定文档的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_name": {
                        "type": "string",
                        "description": "文档名称（如 'report.pdf'）"
                    }
                },
                "required": ["doc_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_docs",
            "description": "列出知识库中已上传的所有文档",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_report",
            "description": "根据检索结果生成结构化报告",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "报告主题"
                    },
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "报告章节列表（可选）"
                    }
                },
                "required": ["topic"]
            }
        }
    }
]


# ReAct 系统提示词
REACT_SYSTEM_PROMPT = """你是一个智能助手，能够使用工具完成用户的任务。

## 可用工具

1. **search_docs**: 从知识库检索文档
2. **summarize_doc**: 总结指定文档
3. **list_docs**: 列出所有已上传文档
4. **generate_report**: 生成结构化报告

## 工作模式

采用 ReAct 模式（推理→行动→观察→循环）：
1. **Thought**: 分析用户任务，决定下一步行动
2. **Action**: 调用合适的工具
3. **Observation**: 观察工具返回结果
4. 重复以上步骤直到完成任务

## 回答要求

- 用中文回答
- 多步任务要分步执行，不要跳步
- 每次工具调用后，根据结果决定是否需要继续
- 最终给出完整、结构清晰的回答
"""


class Agent:
    """AI Agent，支持 ReAct 模式和 Function Calling"""

    def __init__(self, retriever: Optional[Retriever] = None):
        """
        初始化 Agent

        Args:
            retriever: 检索器实例
        """
        self.retriever = retriever or Retriever()
        self.generator = Generator(self.retriever)
        self._client = None
        self._tools_map: Dict[str, Callable] = {
            "search_docs": self._search_docs,
            "summarize_doc": self._summarize_doc,
            "list_docs": self._list_docs,
            "generate_report": self._generate_report,
        }

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI 客户端"""
        if self._client is None:
            validate_config()
            self._client = OpenAI(
                api_key=LLM_API_KEY,
                base_url=LLM_BASE_URL,
            )
        return self._client

    def run(self, query: str, chat_history: Optional[List[dict]] = None) -> dict:
        """
        执行 Agent 任务

        Args:
            query: 用户输入
            chat_history: 对话历史

        Returns:
            {
                "answer": str,          # 最终回答
                "thoughts": list,       # 思考过程
                "tool_calls": list,     # 工具调用记录
            }
        """
        messages = self._build_messages(query, chat_history)
        thoughts = []
        tool_calls = []
        max_iterations = 10  # 防止无限循环

        for _ in range(max_iterations):
            # 调用 LLM
            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # 记录思考过程
            if message.content:
                thoughts.append(message.content)

            # 检查是否有工具调用
            if not message.tool_calls:
                # 没有工具调用，返回最终答案
                return {
                    "answer": message.content,
                    "thoughts": thoughts,
                    "tool_calls": tool_calls,
                }

            # 处理工具调用
            messages.append(message)

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                # 记录工具调用
                tool_calls.append({
                    "name": func_name,
                    "arguments": func_args,
                })

                # 执行工具
                tool_result = self._execute_tool(func_name, func_args)

                # 添加工具结果到消息
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        # 达到最大迭代次数
        return {
            "answer": "任务执行超过最大步数，请尝试简化任务。",
            "thoughts": thoughts,
            "tool_calls": tool_calls,
        }

    def _build_messages(self, query: str, chat_history: Optional[List[dict]] = None) -> List[dict]:
        """构建消息列表"""
        messages = [
            {"role": "system", "content": REACT_SYSTEM_PROMPT}
        ]

        # 添加对话历史
        if chat_history:
            for msg in chat_history[-6:]:  # 最近 3 轮
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # 添加用户输入
        messages.append({"role": "user", "content": query})

        return messages

    def _execute_tool(self, name: str, arguments: dict) -> Any:
        """执行工具"""
        if name not in self._tools_map:
            return {"error": f"未知工具: {name}"}

        try:
            return self._tools_map[name](**arguments)
        except Exception as e:
            return {"error": f"工具执行失败: {str(e)}"}

    def _search_docs(self, query: str, top_k: int = 5) -> dict:
        """检索知识库"""
        # 临时修改 top_k
        original_top_k = self.retriever.top_k
        self.retriever.top_k = top_k

        try:
            nodes = self.retriever.retrieve(query)
            context = format_context(nodes)

            # 提取来源
            sources = []
            for node in nodes:
                sources.append({
                    "source": node.node.metadata.get("source", "未知"),
                    "page": node.node.metadata.get("page", ""),
                    "snippet": node.node.get_content()[:200],
                })

            return {
                "context": context,
                "sources": sources,
                "count": len(nodes),
            }
        finally:
            self.retriever.top_k = original_top_k

    def _summarize_doc(self, doc_name: str) -> dict:
        """总结指定文档"""
        # 先检索该文档的内容
        nodes = self.retriever.retrieve(f"关于 {doc_name} 的内容")

        if not nodes:
            return {"error": f"未找到文档: {doc_name}"}

        # 合并文档内容
        doc_content = "\n\n".join([node.node.get_content() for node in nodes])

        # 调用 LLM 生成摘要
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个文档摘要助手。请用中文生成简洁、结构清晰的文档摘要。"},
                {"role": "user", "content": f"请总结以下文档内容：\n\n{doc_content[:4000]}"}
            ],
        )

        return {
            "doc_name": doc_name,
            "summary": response.choices[0].message.content,
        }

    def _list_docs(self) -> dict:
        """列出所有文档"""
        raw_dir = Path(BASE_DIR) / "data" / "raw"

        if not raw_dir.exists():
            return {"documents": [], "message": "暂无文档"}

        files = list(raw_dir.glob("*"))
        documents = []

        for file in files:
            if file.is_file():
                documents.append({
                    "name": file.name,
                    "size_kb": round(file.stat().st_size / 1024, 1),
                    "type": file.suffix,
                })

        return {
            "documents": documents,
            "count": len(documents),
        }

    def _generate_report(self, topic: str, sections: Optional[List[str]] = None) -> dict:
        """生成结构化报告"""
        # 检索相关文档
        nodes = self.retriever.retrieve(topic)
        context = format_context(nodes)

        # 构建提示词
        sections_prompt = ""
        if sections:
            sections_prompt = f"\n报告应包含以下章节：{', '.join(sections)}"

        prompt = f"""请根据以下参考资料，生成一份关于"{topic}"的结构化报告。

{sections_prompt}

## 要求
- 使用 Markdown 格式
- 包含标题、摘要、正文、结论
- 适当引用数据来源
- 语言简洁专业

## 参考资料

{context}
"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个专业的报告撰写助手。"},
                {"role": "user", "content": prompt}
            ],
        )

        return {
            "topic": topic,
            "report": response.choices[0].message.content,
            "sources": [node.node.metadata.get("source", "未知") for node in nodes],
        }
