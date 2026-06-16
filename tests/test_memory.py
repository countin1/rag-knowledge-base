"""
对话记忆模块测试

测试 ChatMemory 和 SessionManager 类。
"""

import pytest
from backend.memory import ChatMemory, SessionManager


class TestChatMemory:
    """ChatMemory 测试"""

    def test_init(self):
        """测试初始化"""
        memory = ChatMemory(max_history=5)
        assert memory.max_history == 5
        assert memory.is_empty
        assert memory.round_count == 0

    def test_add_messages(self):
        """测试添加消息"""
        memory = ChatMemory()

        memory.add_user_message("你好")
        assert not memory.is_empty
        assert memory.round_count == 0  # 只有用户消息，没有完整轮次

        memory.add_assistant_message("你好！")
        assert memory.round_count == 1

    def test_get_history(self):
        """测试获取历史"""
        memory = ChatMemory()

        memory.add_user_message("问题1")
        memory.add_assistant_message("回答1")
        memory.add_user_message("问题2")
        memory.add_assistant_message("回答2")

        history = memory.get_history()
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "问题1"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "回答1"
        assert history[2]["role"] == "user"
        assert history[2]["content"] == "问题2"
        assert history[3]["role"] == "assistant"
        assert history[3]["content"] == "回答2"

    def test_get_recent(self):
        """测试获取最近对话"""
        memory = ChatMemory()

        # 添加 3 轮对话
        for i in range(3):
            memory.add_user_message(f"问题{i}")
            memory.add_assistant_message(f"回答{i}")

        # 获取最近 1 轮
        recent = memory.get_recent(1)
        assert len(recent) == 2
        assert recent[0]["content"] == "问题2"
        assert recent[1]["content"] == "回答2"

        # 获取最近 2 轮
        recent = memory.get_recent(2)
        assert len(recent) == 4
        assert recent[0]["content"] == "问题1"

    def test_max_history(self):
        """测试最大历史限制"""
        memory = ChatMemory(max_history=2)

        # 添加 3 轮对话
        for i in range(3):
            memory.add_user_message(f"问题{i}")
            memory.add_assistant_message(f"回答{i}")

        # 应该只保留最近 2 轮
        assert memory.round_count == 2
        history = memory.get_history()
        assert len(history) == 4
        assert history[0]["content"] == "问题1"  # 第一轮被保留

    def test_clear(self):
        """测试清空历史"""
        memory = ChatMemory()

        memory.add_user_message("问题")
        memory.add_assistant_message("回答")
        assert not memory.is_empty

        memory.clear()
        assert memory.is_empty
        assert memory.round_count == 0
        assert len(memory.get_history()) == 0


class TestSessionManager:
    """SessionManager 测试"""

    def test_get_session(self):
        """测试获取会话"""
        manager = SessionManager()

        memory1 = manager.get_session("session-1")
        assert isinstance(memory1, ChatMemory)

        # 获取同一会话应该返回相同实例
        memory1_again = manager.get_session("session-1")
        assert memory1 is memory1_again

    def test_different_sessions(self):
        """测试不同会话"""
        manager = SessionManager()

        memory1 = manager.get_session("session-1")
        memory2 = manager.get_session("session-2")

        assert memory1 is not memory2

    def test_list_sessions(self):
        """测试列出会话"""
        manager = SessionManager()

        manager.get_session("session-1")
        manager.get_session("session-2")
        manager.get_session("session-3")

        sessions = manager.list_sessions()
        assert len(sessions) == 3
        assert "session-1" in sessions
        assert "session-2" in sessions
        assert "session-3" in sessions

    def test_delete_session(self):
        """测试删除会话"""
        manager = SessionManager()

        manager.get_session("session-1")
        manager.get_session("session-2")

        assert len(manager.list_sessions()) == 2

        manager.delete_session("session-1")
        sessions = manager.list_sessions()
        assert len(sessions) == 1
        assert "session-1" not in sessions
        assert "session-2" in sessions

    def test_delete_nonexistent_session(self):
        """测试删除不存在的会话"""
        manager = SessionManager()

        # 删除不存在的会话应该不报错
        manager.delete_session("nonexistent")
        assert len(manager.list_sessions()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
