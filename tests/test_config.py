"""
配置模块测试

测试配置加载和验证。
"""

import pytest
import os
from unittest.mock import patch


class TestConfig:
    """配置测试"""

    def test_config_loads(self):
        """测试配置加载"""
        # 设置测试环境变量
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key",
            "LLM_MODEL": "test-model",
            "EMBEDDING_MODEL": "test-embedding",
        }):
            # 重新加载配置模块
            import importlib
            import backend.config
            importlib.reload(backend.config)

            from backend.config import (
                LLM_API_KEY,
                LLM_MODEL,
                EMBEDDING_MODEL,
                CHUNK_SIZE,
                TOP_K,
            )

            assert LLM_API_KEY == "test-key"
            assert LLM_MODEL == "test-model"
            assert EMBEDDING_MODEL == "test-embedding"
            assert isinstance(CHUNK_SIZE, int)
            assert isinstance(TOP_K, int)

    def test_config_defaults(self):
        """测试默认配置值"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key",
        }):
            import importlib
            import backend.config
            importlib.reload(backend.config)

            from backend.config import (
                CHUNK_SIZE,
                CHUNK_OVERLAP,
                TOP_K,
                SIMILARITY_THRESHOLD,
                MAX_HISTORY,
            )

            assert CHUNK_SIZE == 512
            assert CHUNK_OVERLAP == 50
            assert TOP_K == 5
            assert SIMILARITY_THRESHOLD == 0.3
            assert MAX_HISTORY == 10

    def test_config_validation(self):
        """测试配置验证"""
        from backend.config import validate_config

        # 测试缺少必需的 API Key
        with patch.dict(os.environ, {}, clear=True):
            # 清除 DEEPSEEK_API_KEY
            if "DEEPSEEK_API_KEY" in os.environ:
                del os.environ["DEEPSEEK_API_KEY"]

            # 重新加载配置
            import importlib
            import backend.config
            importlib.reload(backend.config)

            # 调用验证函数应该抛出异常
            with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
                validate_config()

    def test_validate_config_success(self):
        """测试配置验证成功"""
        from backend.config import validate_config

        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-key"}):
            import importlib
            import backend.config
            importlib.reload(backend.config)

            # 有 API Key 时应该不抛出异常
            validate_config()

    def test_config_types(self):
        """测试配置类型"""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key",
            "CHUNK_SIZE": "1024",
            "TOP_K": "10",
        }):
            import importlib
            import backend.config
            importlib.reload(backend.config)

            from backend.config import (
                CHUNK_SIZE,
                TOP_K,
            )

            assert isinstance(CHUNK_SIZE, int)
            assert isinstance(TOP_K, int)
            assert CHUNK_SIZE == 1024
            assert TOP_K == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
