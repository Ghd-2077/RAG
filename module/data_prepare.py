"""
数据准备模块（LangChain 实现）
"""

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """数据准备模块 - 负责文档加载、元数据增强和分块。"""

    CATEGORY_MAPPING = {
        "meat_dish": "荤菜",
        "vegetable_dish": "素菜",
        "soup": "汤品",
        "dessert": "甜品",
        "breakfast": "早餐",
        "staple": "主食",
        "aquatic": "水产",
        "condiment": "调料",
        "drink": "饮品",
    }
    CATEGORY_LABELS = sorted(set(CATEGORY_MAPPING.values()))
    DIFFICULTY_LABELS = ["非常简单", "简单", "中等", "困难", "非常困难"]

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.documents: List[Document] = []  # 父文档（完整食谱）
        self.chunks: List[Document] = []  # 子文档（可检索块）
        self.parent_child_map: Dict[str, str] = {}  # child_id -> parent_id

    def load_documents(self) -> List[Document]:
        """
        使用 LangChain DirectoryLoader + TextLoader 递归加载 Markdown 文件。
        """
        data_root = Path(self.data_path)
        if not data_root.exists():
            raise FileNotFoundError(f"数据路径不存在: {self.data_path}")

        logger.info("正在从 %s 加载 Markdown 文档...", data_root)

        loader = DirectoryLoader(
            str(data_root),
            glob="**/*.md",
            loader_cls=TextLoader,#纯文本加载，不解析格式
            loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
            show_progress=False,
            silent_errors=True,
        )
        #存所有Md文件
        raw_docs = loader.load()

        documents: List[Document] = []
        #dange
        for raw_doc in raw_docs:
            #单个md文件
            source = str(raw_doc.metadata.get("source", "")).strip()
            if not source:
                logger.warning("跳过缺少 source 元数据的文档")
                continue

            source_path = Path(source)
            try:
                relative_source = source_path.resolve().relative_to(data_root.resolve()).as_posix()
            except Exception:
                relative_source = source_path.as_posix()

            parent_id = hashlib.md5(relative_source.encode("utf-8")).hexdigest()
            parent_doc = Document(
                page_content=raw_doc.page_content,
                metadata={
                    "source": str(source_path),
                    "relative_source": relative_source,
                    "parent_id": parent_id,
                    "doc_type": "parent",
                },
            )
            self._enhance_metadata(parent_doc)
            documents.append(parent_doc)

        self.documents = documents
        self.parent_child_map.clear()
        logger.info("成功加载 %d 个父文档", len(documents))
        return documents

    def _enhance_metadata(self, doc: Document) -> None:
        """增强文档元数据：分类、菜名、难度。"""
        file_path = Path(doc.metadata.get("relative_source") or doc.metadata.get("source", ""))
        parts_lower = {part.lower() for part in file_path.parts}

        category = "其他"
        for folder_name, label in self.CATEGORY_MAPPING.items():
            if folder_name in parts_lower:
                category = label
                break
        doc.metadata["category"] = category
        doc.metadata["dish_name"] = file_path.stem
        doc.metadata["difficulty"] = self._extract_difficulty(doc.page_content)

    @staticmethod
    def _extract_difficulty(content: str) -> str:
        if "★★★★★" in content:
            return "非常困难"
        if "★★★★" in content:
            return "困难"
        if "★★★" in content:
            return "中等"
        if "★★" in content:
            return "简单"
        if "★" in content:
            return "非常简单"
        return "未知"

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        return cls.CATEGORY_LABELS

    @classmethod
    def get_supported_difficulties(cls) -> List[str]:
        return cls.DIFFICULTY_LABELS

    def chunk_documents(self) -> List[Document]:
        """进行结构化分块并补充块级元数据。"""
        if not self.documents:
            raise ValueError("请先加载文档")

        logger.info("正在进行文档分块...")
        chunks = self._markdown_header_split()

        for i, chunk in enumerate(chunks):
            chunk.metadata.setdefault("chunk_id", str(uuid.uuid4()))
            chunk.metadata["batch_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = chunks
        logger.info("分块完成，共生成 %d 个 chunk", len(chunks))
        return chunks

    def _markdown_header_split(self) -> List[Document]:
        """
        先用 MarkdownHeaderTextSplitter 进行结构化切分；
        如果文档无标题结构，则使用 RecursiveCharacterTextSplitter 兜底。
        """
        #初始化分割器实例，定义切分规则
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
            strip_headers=False,
        )
        #字符递归chunk（备用选项）
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=120,
            separators=["\n### ", "\n## ", "\n\n", "\n", "。", "，", " ", ""],
        )

        all_chunks: List[Document] = []
        self.parent_child_map.clear()

        for doc in self.documents:
            parent_id = doc.metadata.get("parent_id")
            if not parent_id:
                logger.warning("文档缺少 parent_id，跳过: %s", doc.metadata.get("source", "unknown"))
                continue

            try:
                header_chunks = markdown_splitter.split_text(doc.page_content)
                if not header_chunks:
                    header_chunks = [Document(page_content=doc.page_content, metadata={})]
            except Exception as exc:
                logger.warning("Markdown 分割失败，使用兜底分割: %s, error=%s", doc.metadata.get("source"), exc)
                header_chunks = []

            if len(header_chunks) <= 1:
                base_chunks = fallback_splitter.split_documents([doc])
            else:
                base_chunks = header_chunks

            for chunk_index, chunk in enumerate(base_chunks):
                child_id = str(uuid.uuid4())
                merged_metadata = {
                    **doc.metadata,
                    **chunk.metadata,
                    "chunk_id": child_id,
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_index": chunk_index,
                }
                child_doc = Document(page_content=chunk.page_content, metadata=merged_metadata)
                self.parent_child_map[child_id] = parent_id
                all_chunks.append(child_doc)

        logger.info("结构化分块完成，生成 %d 个块", len(all_chunks))
        return all_chunks

    def filter_documents_by_category(self, category: str) -> List[Document]:
        return [doc for doc in self.documents if doc.metadata.get("category") == category]

    def filter_documents_by_difficulty(self, difficulty: str) -> List[Document]:
        return [doc for doc in self.documents if doc.metadata.get("difficulty") == difficulty]

    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {}

        categories: Dict[str, int] = {}
        difficulties: Dict[str, int] = {}
        for doc in self.documents:
            category = doc.metadata.get("category", "未知")
            categories[category] = categories.get(category, 0) + 1

            difficulty = doc.metadata.get("difficulty", "未知")
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1

        avg_chunk_size = (
            sum(chunk.metadata.get("chunk_size", 0) for chunk in self.chunks) / len(self.chunks)
            if self.chunks
            else 0
        )
        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "difficulties": difficulties,
            "avg_chunk_size": avg_chunk_size,
        }

    def export_metadata(self, output_path: str) -> None:
        metadata_list = [
            {
                "source": doc.metadata.get("source"),
                "dish_name": doc.metadata.get("dish_name"),
                "category": doc.metadata.get("category"),
                "difficulty": doc.metadata.get("difficulty"),
                "content_length": len(doc.page_content),
            }
            for doc in self.documents
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)
        logger.info("元数据已导出到: %s", output_path)

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        """
        根据子块结果聚合父文档，并按命中次数降序返回（去重）。
        """
        parent_relevance: Dict[str, int] = {}
        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if parent_id:
                parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1

        parent_docs_map = {
            doc.metadata.get("parent_id"): doc
            for doc in self.documents
            if doc.metadata.get("parent_id")
        }
        sorted_parent_ids = sorted(parent_relevance, key=parent_relevance.get, reverse=True)

        parent_docs: List[Document] = []
        for parent_id in sorted_parent_ids:
            parent_doc = parent_docs_map.get(parent_id)
            if parent_doc is not None:
                parent_docs.append(parent_doc)

        parent_info = [
            f"{doc.metadata.get('dish_name', '未知菜品')}({parent_relevance.get(doc.metadata.get('parent_id', ''), 0)}块)"
            for doc in parent_docs
        ]
        logger.info(
            "从 %d 个子块中找到 %d 个去重父文档: %s",
            len(child_chunks),
            len(parent_docs),
            ", ".join(parent_info),
        )
        return parent_docs
