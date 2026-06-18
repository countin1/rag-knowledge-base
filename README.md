[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

# 📚 RAG Knowledge Base

> 基于 LlamaIndex + ChromaDB + DeepSeek 的企业知识库问答系统 + AI Agent

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
cd rag-knowledge-base

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 文件，填入你的 DeepSeek API Key
DEEPSEEK_API_KEY=your_api_key_here
```

获取 API Key: https://platform.deepseek.com/

### 3. 运行

```bash
streamlit run app.py
```

访问 http://localhost:8501 即可使用。

## 📦 功能特性

### 核心功能

- ✅ **文档上传**: 支持 PDF、TXT、Markdown 格式
- ✅ **智能分块**: 语义分块，保持上下文完整性
- ✅ **向量检索**: 基于 ChromaDB 的高效检索
- ✅ **多轮对话**: 支持上下文记忆的连续问答
- ✅ **答案溯源**: 显示答案来源，支持跳转原文
- ✅ **AI Agent**: 支持多步任务执行，Function Calling

### Agent 功能（新增）

基于 **ReAct 模式**（推理→行动→观察→循环）的智能助手：

- 🤖 **search_docs**: 检索知识库回答问题
- 📝 **summarize_doc**: 总结指定文档内容
- 📄 **list_docs**: 列出已上传的所有文档
- 📊 **generate_report**: 根据检索结果生成结构化报告

**使用场景示例**：
```
用户: 检索四川经济数据，生成摘要，列出要点

Agent 执行流程:
1. Thought: 需要先检索四川经济相关文档
   Action: search_docs("四川经济")
2. Thought: 找到相关文档，需要生成摘要
   Action: generate_report("四川经济数据")
3. 返回结构化报告
```

### 技术亮点

- **LlamaIndex**: RAG 专用框架，API 简洁
- **BGE Embedding**: 中文效果优秀的开源 Embedding 模型
- **ChromaDB**: 轻量级向量数据库，零配置
- **DeepSeek**: 高性价比的国产大模型
- **Function Calling**: DeepSeek 原生工具调用能力

## 🏗️ 项目结构

```
rag-knowledge-base/
├── app.py                  # Streamlit 主界面
├── backend/
│   ├── config.py           # 配置管理
│   ├── ingest.py           # 文档摄取（解析、分块、向量化）
│   ├── retriever.py        # 检索模块
│   ├── generator.py        # LLM 生成模块
│   ├── agent.py            # AI Agent 模块（新增）
│   └── memory.py           # 对话记忆管理
├── data/
│   ├── raw/                # 原始文档存储
│   └── chroma_db/          # 向量库持久化
├── tests/
│   ├── test_ingest.py      # 文档摄取测试
│   ├── test_retriever.py   # 检索测试
│   └── test_integration.py # 集成测试
├── .env.example            # 配置模板
├── requirements.txt        # 依赖清单
└── README.md               # 项目说明
```

## 🔧 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DEEPSEEK_API_KEY` | - | DeepSeek API Key |
| `LLM_MODEL` | deepseek-chat | LLM 模型名称 |
| `EMBEDDING_MODEL` | BAAI/bge-small-zh-v1.5 | Embedding 模型 |
| `CHUNK_SIZE` | 512 | 分块大小（字符数） |
| `CHUNK_OVERLAP` | 50 | 分块重叠（字符数） |
| `TOP_K` | 5 | 检索返回的最大文档数 |
| `SIMILARITY_THRESHOLD` | 0.3 | 相似度阈值 |

## 📊 RAG 流程

```
用户提问
    ↓
向量检索（Top-K）
    ↓
格式化上下文
    ↓
构建 Prompt
    ↓
调用 LLM
    ↓
返回答案 + 来源
```

## 🤖 Agent 工作流程

```
用户任务
    ↓
ReAct 循环（最多 10 轮）
    ├─ Thought: 分析任务，决定行动
    ├─ Action: 调用工具（Function Calling）
    ├─ Observation: 观察工具结果
    └─ 循环直到任务完成
    ↓
返回最终结果 + 执行记录
```

## 🎯 面试话术

> 我做了一个 RAG 企业知识库系统，支持 PDF/TXT/Markdown 文档上传，自动进行语义分块和向量化存储。检索使用 ChromaDB，Embedding 用的是 BGE-small-zh-v1.5。生成部分调用 DeepSeek API，支持多轮对话和答案溯源。
>
> **新增 Agent 功能**：基于 ReAct 模式实现了 AI Agent，支持 Function Calling。定义了 4 个工具：search_docs（检索）、summarize_doc（总结）、list_docs（列出文档）、generate_report（生成报告）。Agent 能够理解多步任务，自动规划执行流程，比如"检索四川经济数据，生成摘要，列出要点"会自动执行检索→总结→提取要点的完整流程。

## 📈 后续优化

- [ ] 混合检索（BM25 + 向量检索）
- [ ] 重排序（BGE Reranker）
- [ ] RAG 评估体系（Ragas）
- [x] Agent 功能（工具调用）✅
- [ ] 知识图谱集成

## ☁️ 部署到 Streamlit Cloud

1. 将代码推送到 GitHub 仓库
2. 去 [share.streamlit.io](https://share.streamlit.io) 连接仓库并部署
3. 在 App settings -> Secrets 中添加：

```toml
DEEPSEEK_API_KEY = "your-deepseek-key"
```

Streamlit Cloud 会自动将 Secrets 中的键暴露为环境变量。

## 📄 License

MIT
