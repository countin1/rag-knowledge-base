"""
检索模块测试
"""

import pytest
from unittest.mock import Mock, patch

from backend.retriever import Retriever, format_context


class TestRetriever:
    """检索器测试"""

    def test_retriever_init(self):
        """测试检索器初始化"""
        retriever = Retriever(top_k=3)
        assert retriever.top_k == 3

    @patch("backend.retriever.get_index")
    def test_retrieve_returns_nodes(self, mock_get_index):
        """测试检索返回节点"""
        # Mock 索引和检索器
        mock_retriever = Mock()
        mock_node = Mock()
        mock_node.score = 0.8
        mock_node.node.get_content.return_value = "测试内容"
        mock_node.node.metadata = {"source": "test.txt"}
        mock_retriever.retrieve.return_value = [mock_node]

        mock_index = Mock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_get_index.return_value = mock_index

        retriever = Retriever(top_k=5)
        results = retriever.retrieve("测试查询")

        assert len(results) == 1
        assert results[0].score == 0.8


class TestFormatContext:
    """上下文格式化测试"""

    def test_format_empty_nodes(self):
        """测试空节点列表"""
        result = format_context([])
        assert "未找到相关文档" in result

    def test_format_single_node(self):
        """测试单个节点"""
        mock_node = Mock()
        mock_node.score = 0.9
        mock_node.node.get_content.return_value = "这是测试内容"
        mock_node.node.metadata = {"source": "test.txt", "page": 1}

        result = format_context([mock_node])
        assert "test.txt" in result
        assert "第 1 页" in result
        assert "0.90" in result
        assert "这是测试内容" in result

    def test_format_multiple_nodes(self):
        """测试多个节点"""
        nodes = []
        for i in range(3):
            mock_node = Mock()
            mock_node.score = 0.8 - i * 0.1
            mock_node.node.get_content.return_value = f"内容 {i}"
            mock_node.node.metadata = {"source": f"file{i}.txt"}
            nodes.append(mock_node)

        result = format_context(nodes)
        assert "---" in result  # 分隔符
        assert "file0.txt" in result
        assert "file1.txt" in result
        assert "file2.txt" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
