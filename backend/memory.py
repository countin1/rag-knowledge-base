"""
对话记忆模块

管理多轮对话历史。
"""

from typing import List, Optional
from collections import deque

from .config import MAX_HISTORY


class ChatMemory:
    """对话记忆"""

    def __init__(self, max_history: int = MAX_HISTORY):
        """
        初始化对话记忆

        Args:
            max_history: 最大历史轮数
        """
        self.max_history = max_history
        self._history: deque = deque(maxlen=max_history * 2)  # 每轮包含 user + assistant

    def add_user_message(self, content: str) -> None:
        """添加用户消息"""
        self._history.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """添加助手消息"""
        self._history.append({"role": "assistant", "content": content})

    def get_history(self) -> List[dict]:
        """获取完整对话历史"""
        return list(self._history)

    def get_recent(self, n_rounds: int = 3) -> List[dict]:
        """获取最近 n 轮对话"""
        return list(self._history)[-n_rounds * 2:]

    def clear(self) -> None:
        """清空对话历史"""
        self._history.clear()

    @property
    def is_empty(self) -> bool:
        """是否为空"""
        return len(self._history) == 0

    @property
    def round_count(self) -> int:
        """对话轮数"""
        return len(self._history) // 2


class SessionManager:
    """会话管理器"""

    def __init__(self):
        self._sessions: dict = {}

    def get_session(self, session_id: str) -> ChatMemory:
        """
        获取或创建会话

        Args:
            session_id: 会话 ID

        Returns:
            对话记忆实例
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatMemory()
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> List[str]:
        """列出所有会话 ID"""
        return list(self._sessions.keys())


# 全局会话管理器
session_manager = SessionManager()
