"""
Pytest 配置文件

配置测试环境和 fixtures。
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """配置 pytest"""
    config.addinivalue_line(
        "markers",
        "integration: 集成测试（需要外部依赖）"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    # 如果没有安装 llama_index，跳过需要它的测试
    try:
        import llama_index
    except ImportError:
        skip_integration = pytest.mark.skip(
            reason="llama_index 未安装，跳过集成测试"
        )
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def project_root():
    """项目根目录"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """测试数据目录"""
    test_dir = project_root / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir
