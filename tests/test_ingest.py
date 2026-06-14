"""
文档摄取模块测试
"""

import pytest
import tempfile
import os
from pathlib import Path

from backend.ingest import parse_document, get_embedding_model, get_vector_store


class TestDocumentParsing:
    """文档解析测试"""

    def test_parse_txt(self):
        """测试 TXT 文件解析"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("这是一个测试文档。\n包含多行内容。")
            temp_path = f.name

        try:
            docs = parse_document(temp_path)
            assert len(docs) == 1
            assert "测试文档" in docs[0].text
            assert docs[0].metadata["source"] == temp_path
        finally:
            os.unlink(temp_path)

    def test_parse_md(self):
        """测试 Markdown 文件解析"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write("# 测试标题\n\n这是 Markdown 内容。")
            temp_path = f.name

        try:
            docs = parse_document(temp_path)
            assert len(docs) == 1
            assert "测试标题" in docs[0].text
        finally:
            os.unlink(temp_path)

    def test_unsupported_format(self):
        """测试不支持的文件格式"""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="不支持的文件格式"):
                parse_document(temp_path)
        finally:
            os.unlink(temp_path)


class TestEmbeddingModel:
    """Embedding 模型测试"""

    def test_get_embedding_model(self):
        """测试获取 Embedding 模型"""
        model = get_embedding_model()
        assert model is not None

    def test_embedding_dimensions(self):
        """测试 Embedding 维度"""
        model = get_embedding_model()
        embedding = model.get_text_embedding("测试文本")
        assert len(embedding) == 512  # BGE-small 维度


class TestVectorStore:
    """向量存储测试"""

    def test_get_vector_store(self):
        """测试获取向量存储"""
        store = get_vector_store()
        assert store is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
