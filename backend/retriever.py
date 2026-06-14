"""
检索模块

负责从向量库中检索相关文档。
"""

from typing import List, Tuple

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

from .ingest import get_index
from .config import TOP_K, SIMILARITY_THRESHOLD


class Retriever:
    """检索器"""

    def __init__(self, top_k: int = TOP_K):
        """
        初始化检索器

        Args:
            top_k: 返回的最大结果数
        """
        self.top_k = top_k
        self._index = None

    @property
    def index(self) -> VectorStoreIndex:
        """懒加载索引"""
        if self._index is None:
            self._index = get_index()
        return self._index

    def retrieve(self, query: str) -> List[NodeWithScore]:
        """
        检索相关文档

        Args:
            query: 查询文本

        Returns:
            相关文档节点列表
        """
        retriever = self.index.as_retriever(
            similarity_top_k=self.top_k,
        )

        nodes = retriever.retrieve(query)

        # 过滤低相似度结果
        filtered = [
            node for node in nodes
            if node.score is None or node.score >= SIMILARITY_THRESHOLD
        ]

        return filtered

    def retrieve_with_scores(self, query: str) -> List[Tuple[str, float, dict]]:
        """
        检索并返回带分数的结果

        Args:
            query: 查询文本

        Returns:
            (文本, 分数, 元数据) 列表
        """
        nodes = self.retrieve(query)

        results = []
        for node in nodes:
            results.append((
                node.node.get_content(),
                node.score or 0.0,
                node.node.metadata or {},
            ))

        return results


def format_context(nodes: List[NodeWithScore]) -> str:
    """
    格式化检索结果为上下文

    Args:
        nodes: 检索结果节点

    Returns:
        格式化的上下文文本
    """
    if not nodes:
        return "未找到相关文档。"

    context_parts = []
    for i, node in enumerate(nodes, 1):
        source = node.node.metadata.get("source", "未知来源")
        page = node.node.metadata.get("page", "")
        score = node.score or 0.0

        header = f"[文档 {i}] 来源: {source}"
        if page:
            header += f", 第 {page} 页"
        header += f" (相似度: {score:.2f})"

        context_parts.append(f"{header}\n{node.node.get_content()}")

    return "\n\n---\n\n".join(context_parts)
