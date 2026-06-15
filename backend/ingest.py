"""
文档摄取模块

负责文档上传、解析、分块和向量化。
"""

import os
from pathlib import Path
from typing import List, Optional

from llama_index.core import Document, VectorStoreIndex, Settings, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def get_embedding_model() -> HuggingFaceEmbedding:
    """获取 Embedding 模型"""
    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL,
        device=EMBEDDING_DEVICE,
    )


def get_vector_store() -> ChromaVectorStore:
    """获取向量存储"""
    # 确保目录存在
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)

    # 初始化 ChromaDB
    db = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = db.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return ChromaVectorStore(chroma_collection=collection)


def parse_document(file_path: str) -> List[Document]:
    """
    解析文档

    支持 PDF、TXT、Markdown 格式。

    Args:
        file_path: 文件路径

    Returns:
        文档列表
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(str(path))
    elif suffix in (".txt", ".md"):
        return _parse_text(str(path))
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def _parse_pdf(file_path: str) -> List[Document]:
    """解析 PDF"""
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source": file_path,
                        "page": i + 1,
                        "total_pages": len(reader.pages),
                    },
                )
            )

    return documents


def _parse_text(file_path: str) -> List[Document]:
    """解析纯文本/Markdown"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [
        Document(
            text=text,
            metadata={"source": file_path},
        )
    ]


def ingest_document(
    file_path: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    """
    摄取文档到向量库

    Args:
        file_path: 文件路径
        chunk_size: 分块大小
        chunk_overlap: 分块重叠

    Returns:
        处理的文档块数量
    """
    # 1. 解析文档
    documents = parse_document(file_path)
    print(f"解析文档: {len(documents)} 页")

    # 2. 配置分块器
    node_parser = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # 3. 获取 Embedding 和向量存储
    embed_model = get_embedding_model()
    vector_store = get_vector_store()

    # 4. 配置 Settings
    Settings.embed_model = embed_model
    Settings.node_parser = node_parser

    # 5. 创建 StorageContext 和索引
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    # 统计分块数量
    nodes = node_parser.get_nodes_from_documents(documents)
    print(f"文档摄取完成: {len(nodes)} 个分块")

    return len(nodes)


def get_index() -> VectorStoreIndex:
    """获取已有的向量索引"""
    embed_model = get_embedding_model()
    vector_store = get_vector_store()

    Settings.embed_model = embed_model

    return VectorStoreIndex.from_vector_store(vector_store)
