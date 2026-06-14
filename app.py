"""
RAG Knowledge Base - Streamlit 主界面

企业知识库问答系统
"""

import streamlit as st
import os
import uuid
from pathlib import Path

from backend.ingest import ingest_document, get_index
from backend.retriever import Retriever
from backend.generator import Generator
from backend.memory import session_manager
from backend.config import CHROMA_PERSIST_DIR

# 页面配置
st.set_page_config(
    page_title="RAG 知识库",
    page_icon="📚",
    layout="wide",
)

# 初始化 session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "generator" not in st.session_state:
    st.session_state.generator = Generator()

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True


def main():
    """主函数"""
    st.title("📚 RAG 知识库问答系统")
    st.caption("基于 LlamaIndex + ChromaDB + DeepSeek")

    # 侧边栏
    with st.sidebar:
        sidebar_content()

    # 主界面标签页
    tab1, tab2, tab3 = st.tabs(["💬 对话", "📄 文档管理", "📊 系统信息"])

    with tab1:
        chat_tab()

    with tab2:
        document_tab()

    with tab3:
        system_tab()


def sidebar_content():
    """侧边栏内容"""
    st.header("⚙️ 设置")

    # 会话管理
    st.subheader("会话")
    memory = session_manager.get_session(st.session_state.session_id)

    if st.button("🗑️ 清空对话"):
        memory.clear()
        st.rerun()

    st.caption(f"当前对话轮数: {memory.round_count}")

    # 检索设置
    st.subheader("检索参数")
    top_k = st.slider("返回文档数", 1, 10, 5)
    show_sources = st.checkbox("显示来源", value=True)
    st.session_state.show_sources = show_sources

    # 系统状态
    st.subheader("系统状态")
    chroma_dir = Path(CHROMA_PERSIST_DIR)
    if chroma_dir.exists():
        st.success("✅ 向量库已初始化")
    else:
        st.warning("⚠️ 向量库未初始化")

    # API 配置提示
    st.subheader("配置说明")
    st.info("""
    请在 `.env` 文件中配置:

    ```
    DEEPSEEK_API_KEY=your_key
    ```

    或设置环境变量。
    """)


def chat_tab():
    """对话标签页"""
    memory = session_manager.get_session(st.session_state.session_id)

    # 显示对话历史
    for msg in memory.get_history():
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # 用户输入
    if query := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        with st.chat_message("user"):
            st.write(query)

        # 添加到历史
        memory.add_user_message(query)

        # 生成答案
        with st.chat_message("assistant"):
            with st.spinner("正在检索和生成答案..."):
                result = st.session_state.generator.generate(
                    query=query,
                    chat_history=memory.get_recent(3),
                )

            # 显示答案
            st.write(result["answer"])

            # 显示来源
            if st.session_state.show_sources and result["sources"]:
                with st.expander("📎 引用来源"):
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"""
                        **[{i}] {source['source']}**
                        - 页码: {source['page'] or 'N/A'}
                        - 相似度: {source['score']:.2f}
                        - 片段: {source['snippet']}
                        """)

        # 添加到历史
        memory.add_assistant_message(result["answer"])


def document_tab():
    """文档管理标签页"""
    st.header("📄 文档管理")

    # 文档上传
    st.subheader("上传文档")
    uploaded_files = st.file_uploader(
        "选择文件",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("📥 开始处理"):
            progress = st.progress(0)
            status = st.status("处理中...")

            for i, file in enumerate(uploaded_files):
                status.write(f"处理: {file.name}")

                # 保存临时文件
                temp_dir = Path("data/raw")
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_path = temp_dir / file.name

                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())

                try:
                    # 摄取文档
                    num_chunks = ingest_document(str(temp_path))
                    status.write(f"✅ {file.name}: {num_chunks} 个分块")
                except Exception as e:
                    status.write(f"❌ {file.name}: {e}")

                progress.progress((i + 1) / len(uploaded_files))

            status.update(label="处理完成!", state="complete")
            st.success(f"成功处理 {len(uploaded_files)} 个文档")

    # 已上传文档列表
    st.subheader("已上传文档")
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        files = list(raw_dir.glob("*"))
        if files:
            for file in files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file.name}")
                with col2:
                    st.caption(f"{file.stat().st_size / 1024:.1f} KB")
                with col3:
                    if st.button("🗑️", key=f"del_{file.name}"):
                        file.unlink()
                        st.rerun()
        else:
            st.info("暂无文档")
    else:
        st.info("暂无文档")


def system_tab():
    """系统信息标签页"""
    st.header("📊 系统信息")

    # 配置信息
    st.subheader("当前配置")
    from backend import config

    col1, col2 = st.columns(2)

    with col1:
        st.write("**LLM 配置**")
        st.write(f"- 模型: {config.LLM_MODEL}")
        st.write(f"- API 地址: {config.LLM_BASE_URL}")

    with col2:
        st.write("**Embedding 配置**")
        st.write(f"- 模型: {config.EMBEDDING_MODEL}")
        st.write(f"- 设备: {config.EMBEDDING_DEVICE}")

    st.write("**检索配置**")
    st.write(f"- 分块大小: {config.CHUNK_SIZE}")
    st.write(f"- 分块重叠: {config.CHUNK_OVERLAP}")
    st.write(f"- Top-K: {config.TOP_K}")
    st.write(f"- 相似度阈值: {config.SIMILARITY_THRESHOLD}")

    # 技术栈
    st.subheader("技术栈")
    st.write("""
    | 组件 | 技术 |
    |------|------|
    | 框架 | LlamaIndex |
    | 向量库 | ChromaDB |
    | Embedding | BGE-small-zh-v1.5 |
    | LLM | DeepSeek |
    | 前端 | Streamlit |
    """)

    # 项目结构
    st.subheader("项目结构")
    st.code("""
    rag-knowledge-base/
    ├── app.py              # Streamlit 主界面
    ├── backend/
    │   ├── config.py       # 配置管理
    │   ├── ingest.py       # 文档摄取
    │   ├── retriever.py    # 检索模块
    │   ├── generator.py    # 生成模块
    │   └── memory.py       # 对话记忆
    ├── data/
    │   ├── raw/            # 原始文档
    │   └── chroma_db/      # 向量库
    └── tests/              # 测试
    """)


if __name__ == "__main__":
    main()
