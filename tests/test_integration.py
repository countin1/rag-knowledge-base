"""
集成测试
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

from backend.ingest import ingest_document
from backend.retriever import Retriever
from backend.generator import Generator
from backend.memory import ChatMemory, SessionManager


class TestDocumentIngestion:
    """文档摄取集成测试"""

    @pytest.fixture
    def sample_txt(self):
        """创建测试文档"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("""
            GDP 是国内生产总值（Gross Domestic Product）的简称。
            它是衡量一个国家或地区经济状况和发展水平的重要指标。
            GDP 有三种计算方法：生产法、收入法和支出法。
            """)
            temp_path = f.name

        yield temp_path
        os.unlink(temp_path)

    def test_ingest_txt_file(self, sample_txt):
        """测试摄取 TXT 文件"""
        num_chunks = ingest_document(sample_txt, chunk_size=100, chunk_overlap=20)
        assert num_chunks > 0
        print(f"摄取了 {num_chunks} 个分块")


class TestRetrievalIntegration:
    """检索集成测试"""

    def test_retriever_with_real_data(self):
        """测试检索器（需要先摄取数据）"""
        retriever = Retriever(top_k=3)

        # 这个测试需要先有数据
        # 在实际环境中，先运行 test_ingest_txt_file
        try:
            results = retriever.retrieve_with_scores("什么是 GDP？")
            print(f"检索到 {len(results)} 个结果")
            for text, score, meta in results:
                print(f"  - 相似度: {score:.2f}, 来源: {meta.get('source', 'N/A')}")
        except Exception as e:
            pytest.skip(f"需要先摄取数据: {e}")


class TestMemoryIntegration:
    """对话记忆集成测试"""

    def test_chat_memory_round_trip(self):
        """测试对话记忆完整流程"""
        memory = ChatMemory(max_history=5)

        # 添加对话
        memory.add_user_message("什么是 GDP？")
        memory.add_assistant_message("GDP 是国内生产总值...")

        memory.add_user_message("怎么计算？")
        memory.add_assistant_message("GDP 有三种计算方法...")

        # 验证历史
        assert memory.round_count == 2
        assert not memory.is_empty

        history = memory.get_history()
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

        # 验证最近对话
        recent = memory.get_recent(1)
        assert len(recent) == 2
        assert recent[0]["content"] == "怎么计算？"

        # 清空历史
        memory.clear()
        assert memory.is_empty
        assert memory.round_count == 0

    def test_session_manager(self):
        """测试会话管理器"""
        manager = SessionManager()

        # 创建会话
        memory1 = manager.get_session("session-1")
        memory2 = manager.get_session("session-2")

        assert memory1 is not memory2

        # 获取同一会话
        memory1_again = manager.get_session("session-1")
        assert memory1 is memory1_again

        # 列出会话
        sessions = manager.list_sessions()
        assert "session-1" in sessions
        assert "session-2" in sessions

        # 删除会话
        manager.delete_session("session-1")
        sessions = manager.list_sessions()
        assert "session-1" not in sessions


class TestGeneratorIntegration:
    """生成器集成测试"""

    def test_generator_build_prompt(self):
        """测试生成器构建提示词"""
        generator = Generator()

        # 测试构建提示词
        prompt = generator._build_prompt(
            query="什么是 GDP？",
            context="[文档 1] 来源: test.txt\nGDP 是国内生产总值...",
            chat_history=[
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "你好！"},
            ],
        )

        assert "什么是 GDP？" in prompt
        assert "GDP 是国内生产总值" in prompt
        assert "你好" in prompt
        assert "参考文档" in prompt

    def test_generator_extract_sources(self):
        """测试来源提取"""
        generator = Generator()

        # Mock 节点
        nodes = []
        for i in range(3):
            mock_node = Mock()
            mock_node.score = 0.9 - i * 0.1
            mock_node.node.get_content.return_value = f"内容 {i} " * 50
            mock_node.node.metadata = {"source": f"file{i}.txt", "page": i + 1}
            nodes.append(mock_node)

        sources = generator._extract_sources(nodes)
        assert len(sources) == 3
        assert sources[0]["source"] == "file0.txt"
        assert sources[0]["page"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
