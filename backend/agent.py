"""
Agent 模块

实现基于 ReAct 模式的 AI Agent，支持 Function Calling。
"""

import json
import logging
from typing import List, Dict, Any, Optional, Callable
from openai import OpenAI

from .config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, LLM_MAX_TOKENS, validate_config
from .retriever import Retriever, format_context
from .generator import Generator
from pathlib import Path
from .config import BASE_DIR

logger = logging.getLogger(__name__)


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
    },
    {
        "type": "function",
        "function": {
            "name": "compare_docs",
            "description": "对比多个文档的异同，找出共同点和差异",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要对比的文档名称列表"
                    },
                    "aspect": {
                        "type": "string",
                        "description": "对比维度（如'内容'、'观点'、'数据'等）",
                        "default": "内容"
                    }
                },
                "required": ["doc_names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": "从文档中提取关键实体（人名、地点、时间、数据等）",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_name": {
                        "type": "string",
                        "description": "文档名称"
                    },
                    "entity_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要提取的实体类型（如['人名', '地点', '时间', '数据']）",
                        "default": ["人名", "地点", "时间", "数据"]
                    }
                },
                "required": ["doc_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "classify_docs",
            "description": "自动分类文档，按主题或类型归类",
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "string",
                        "description": "分类标准（如'主题'、'类型'、'时间'）",
                        "default": "主题"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_outline",
            "description": "生成文档大纲或思维导图结构",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_name": {
                        "type": "string",
                        "description": "文档名称"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "大纲深度（1-3），默认为 2",
                        "default": 2
                    }
                },
                "required": ["doc_name"]
            }
        }
    }
]


# ReAct 系统提示词
REACT_SYSTEM_PROMPT = """你是一个智能助手，能够使用工具完成用户的任务。

## 可用工具

1. **search_docs**: 从知识库检索文档，用于回答问题
2. **summarize_doc**: 总结指定文档的内容
3. **list_docs**: 列出所有已上传文档
4. **generate_report**: 生成结构化报告
5. **compare_docs**: 对比多个文档的异同
6. **extract_entities**: 提取文档中的关键实体（人名、地点、时间、数据等）
7. **classify_docs**: 自动分类文档
8. **generate_outline**: 生成文档大纲或思维导图结构

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
            "compare_docs": self._compare_docs,
            "extract_entities": self._extract_entities,
            "classify_docs": self._classify_docs,
            "generate_outline": self._generate_outline,
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
        logger.info(f"Agent 收到任务: {query[:50]}...")

        messages = self._build_messages(query, chat_history)
        thoughts = []
        tool_calls = []
        max_iterations = 10  # 防止无限循环

        for iteration in range(max_iterations):
            # 裁剪消息，防止超出 token 限制
            if len(messages) > 20:
                messages = messages[:1] + messages[-10:]  # 保留 system + 最近 10 条

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
                logger.info(f"任务完成，调用了 {len(tool_calls)} 个工具，迭代 {iteration + 1} 次")
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
        logger.warning(f"Agent 达到最大迭代次数 {max_iterations}")
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
        except ValueError as e:
            # 业务异常：参数错误等
            return {"error": f"参数错误: {str(e)}"}
        except Exception as e:
            # 系统异常：记录日志，返回通用错误
            logger.exception(f"工具 {name} 执行异常")
            return {"error": "内部错误，请稍后重试"}

    def _search_docs(self, query: str, top_k: int = 5) -> dict:
        """检索知识库"""
        # 创建独立的 retriever 实例，避免并发安全问题
        temp_retriever = Retriever(top_k=top_k)
        nodes = temp_retriever.retrieve(query)
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

    def _summarize_doc(self, doc_name: str) -> dict:
        """总结指定文档"""
        # 先检索该文档的内容
        nodes = self.retriever.retrieve(f"关于 {doc_name} 的内容")

        if not nodes:
            return {"error": f"未找到文档: {doc_name}"}

        # 合并文档内容，根据模型上下文动态截断
        doc_content = "\n\n".join([node.node.get_content() for node in nodes])
        max_context = LLM_MAX_TOKENS * 3  # 粗略估算：token 数 * 3 ≈ 字符数
        doc_content = doc_content[:max_context]

        # 调用 LLM 生成摘要
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个文档摘要助手。请用中文生成简洁、结构清晰的文档摘要。"},
                {"role": "user", "content": f"请总结以下文档内容：\n\n{doc_content}"}
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

    def _compare_docs(self, doc_names: List[str], aspect: str = "内容") -> dict:
        """对比多个文档的异同"""
        if len(doc_names) < 2:
            return {"error": "至少需要两个文档进行对比"}

        # 检索每个文档的内容
        docs_content = {}
        for doc_name in doc_names:
            nodes = self.retriever.retrieve(f"关于 {doc_name} 的内容")
            if nodes:
                docs_content[doc_name] = "\n\n".join([node.node.get_content() for node in nodes[:3]])
            else:
                docs_content[doc_name] = f"未找到文档: {doc_name}"

        # 构建对比提示词
        docs_text = "\n\n---\n\n".join([f"## {name}\n{content}" for name, content in docs_content.items()])

        prompt = f"""请从"{aspect}"维度对比以下文档，找出共同点和差异。

{docs_text}

## 输出格式
1. **共同点**: 列出各文档的共同特征
2. **差异点**: 列出各文档的主要区别
3. **总结**: 简要概括对比结论
"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个文档分析专家，擅长对比分析不同文档。"},
                {"role": "user", "content": prompt}
            ],
        )

        return {
            "docs": doc_names,
            "aspect": aspect,
            "comparison": response.choices[0].message.content,
        }

    def _extract_entities(self, doc_name: str, entity_types: Optional[List[str]] = None) -> dict:
        """提取文档中的关键实体"""
        if entity_types is None:
            entity_types = ["人名", "地点", "时间", "数据"]

        # 检索文档内容
        nodes = self.retriever.retrieve(f"关于 {doc_name} 的内容")
        if not nodes:
            return {"error": f"未找到文档: {doc_name}"}

        doc_content = "\n\n".join([node.node.get_content() for node in nodes[:3]])
        max_context = LLM_MAX_TOKENS * 3
        doc_content = doc_content[:max_context]

        # 构建提取提示词
        types_str = "、".join(entity_types)
        prompt = f"""请从以下文档中提取{types_str}等关键实体。

## 文档内容

{doc_content}

## 输出格式

按类别列出提取的实体，每个实体附带简要上下文说明。
"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个信息提取专家，擅长从文本中提取关键实体。"},
                {"role": "user", "content": prompt}
            ],
        )

        return {
            "doc_name": doc_name,
            "entity_types": entity_types,
            "entities": response.choices[0].message.content,
        }

    def _classify_docs(self, criteria: str = "主题") -> dict:
        """自动分类文档"""
        # 获取所有文档
        raw_dir = Path(BASE_DIR) / "data" / "raw"
        if not raw_dir.exists():
            return {"error": "暂无文档"}

        files = list(raw_dir.glob("*"))
        if not files:
            return {"error": "暂无文档"}

        # 检索每个文档的内容摘要
        docs_summary = []
        for file in files[:10]:  # 最多处理 10 个文档
            nodes = self.retriever.retrieve(f"关于 {file.name} 的内容")
            if nodes:
                summary = nodes[0].node.get_content()[:200]
                docs_summary.append({"name": file.name, "summary": summary})

        # 构建分类提示词
        docs_text = "\n".join([f"- **{d['name']}**: {d['summary']}" for d in docs_summary])

        prompt = f"""请根据"{criteria}"标准对以下文档进行分类。

## 文档列表

{docs_text}

## 输出格式

1. **分类结果**: 按类别列出文档
2. **分类说明**: 解释分类依据
"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个文档管理专家，擅长对文档进行分类整理。"},
                {"role": "user", "content": prompt}
            ],
        )

        return {
            "criteria": criteria,
            "classification": response.choices[0].message.content,
            "doc_count": len(docs_summary),
        }

    def _generate_outline(self, doc_name: str, depth: int = 2) -> dict:
        """生成文档大纲或思维导图结构"""
        # 检索文档内容
        nodes = self.retriever.retrieve(f"关于 {doc_name} 的内容")
        if not nodes:
            return {"error": f"未找到文档: {doc_name}"}

        doc_content = "\n\n".join([node.node.get_content() for node in nodes[:5]])
        max_context = LLM_MAX_TOKENS * 3
        doc_content = doc_content[:max_context]

        # 构建大纲提示词
        prompt = f"""请为以下文档生成一个{depth}层深度的大纲结构。

## 文档内容

{doc_content}

## 输出格式

使用 Markdown 缩进列表格式，展示文档的层级结构。
"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个文档结构分析专家，擅长提取文档的逻辑结构。"},
                {"role": "user", "content": prompt}
            ],
        )

        return {
            "doc_name": doc_name,
            "depth": depth,
            "outline": response.choices[0].message.content,
        }
