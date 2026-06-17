"""
生成模块

负责调用 LLM 生成答案。
"""

from typing import List, Optional

from llama_index.core import Settings
from llama_index.llms.deepseek import DeepSeek

from .config import LLM_API_KEY, LLM_MODEL, LLM_BASE_URL, LLM_MAX_TOKENS, LLM_TEMPERATURE, validate_config
from .retriever import Retriever, format_context


# 系统提示词
SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据提供的文档内容回答用户的问题。

回答要求：
1. 只基于提供的文档内容回答，不要编造信息
2. 如果文档中没有相关内容，请明确说明
3. 回答要准确、完整、结构清晰
4. 适当引用文档来源

请用中文回答。"""


class Generator:
    """答案生成器"""

    def __init__(self, retriever: Optional[Retriever] = None):
        """
        初始化生成器

        Args:
            retriever: 检索器实例
        """
        self.retriever = retriever or Retriever()
        self._llm = None

    @property
    def llm(self) -> DeepSeek:
        """懒加载 LLM"""
        if self._llm is None:
            validate_config()
            self._llm = DeepSeek(
                model=LLM_MODEL,
                api_key=LLM_API_KEY,
                api_base=LLM_BASE_URL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        return self._llm

    def generate(self, query: str, chat_history: Optional[List[dict]] = None) -> dict:
        """
        生成答案

        Args:
            query: 用户问题
            chat_history: 对话历史

        Returns:
            {
                "answer": str,      # 生成的答案
                "sources": list,    # 引用来源
                "context": str,     # 检索到的上下文
            }
        """
        # 1. 检索相关文档
        nodes = self.retriever.retrieve(query)
        context = format_context(nodes)

        # 2. 构建提示词
        prompt = self._build_prompt(query, context, chat_history)

        # 3. 调用 LLM
        response = self.llm.complete(prompt)
        answer = response.text

        # 4. 提取来源信息
        sources = self._extract_sources(nodes)

        return {
            "answer": answer,
            "sources": sources,
            "context": context,
        }

    def _build_prompt(
        self,
        query: str,
        context: str,
        chat_history: Optional[List[dict]] = None,
    ) -> str:
        """构建完整提示词"""
        parts = [SYSTEM_PROMPT, ""]

        # 添加对话历史
        if chat_history:
            parts.append("对话历史:")
            for msg in chat_history[-6:]:  # 最近 3 轮
                role = "用户" if msg["role"] == "user" else "助手"
                parts.append(f"{role}: {msg['content']}")
            parts.append("")

        # 添加检索上下文
        parts.append("参考文档:")
        parts.append(context)
        parts.append("")

        # 添加用户问题
        parts.append(f"用户问题: {query}")
        parts.append("")
        parts.append("请基于以上文档内容回答:")

        return "\n".join(parts)

    def _extract_sources(self, nodes) -> List[dict]:
        """提取来源信息"""
        sources = []
        seen = set()

        for node in nodes:
            source = node.node.metadata.get("source", "未知来源")
            page = node.node.metadata.get("page", "")

            key = f"{source}:{page}"
            if key not in seen:
                seen.add(key)
                sources.append({
                    "source": source,
                    "page": page,
                    "score": node.score or 0.0,
                    "snippet": node.node.get_content()[:200] + "...",
                })

        return sources
