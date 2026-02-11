import hashlib
import json
from pathlib import Path

import pytest

# 按你的实际文件改这里
from module.data_prepare import DataPreparationModule


def write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_load_documents_and_metadata(tmp_path: Path):
    write_file(
        tmp_path / "meat_dish" / "红烧肉.md",
        "# 红烧肉\n难度：★★★\n内容",
    )

    m = DataPreparationModule(str(tmp_path))
    docs = m.load_documents()

    assert len(docs) == 1
    d = docs[0]
    assert d.metadata["doc_type"] == "parent"
    assert d.metadata["category"] == "荤菜"
    assert d.metadata["dish_name"] == "红烧肉"
    assert d.metadata["difficulty"] == "中等"

    rel = "meat_dish/红烧肉.md"
    assert d.metadata["parent_id"] == hashlib.md5(rel.encode("utf-8")).hexdigest()


def test_chunk_documents_with_markdown_headers(tmp_path: Path):
    write_file(
        tmp_path / "soup" / "番茄蛋汤.md",
        "# 番茄蛋汤\n## 食材\n鸡蛋\n## 做法\n步骤1\n步骤2",
    )

    m = DataPreparationModule(str(tmp_path))
    m.load_documents()
    chunks = m.chunk_documents()

    assert len(chunks) >= 2
    assert all(c.metadata["doc_type"] == "child" for c in chunks)
    assert all("chunk_id" in c.metadata for c in chunks)
    assert all("parent_id" in c.metadata for c in chunks)
    assert len(m.parent_child_map) == len(chunks)


def test_chunk_documents_fallback_when_no_headers(tmp_path: Path):
    long_text = "这是一段没有标题的文本。" * 300
    write_file(tmp_path / "dessert" / "布丁.md", long_text)

    m = DataPreparationModule(str(tmp_path))
    m.load_documents()
    chunks = m.chunk_documents()

    # 无标题时应至少能切出多个块（因为文本足够长）
    assert len(chunks) > 1


def test_get_parent_documents_order(tmp_path: Path):
    write_file(tmp_path / "meat_dish" / "a.md", "# A\n## x\n1\n## y\n2")
    write_file(tmp_path / "soup" / "b.md", "# B\n## x\n1\n## y\n2")

    m = DataPreparationModule(str(tmp_path))
    m.load_documents()
    chunks = m.chunk_documents()

    # 人工构造：让 A 命中 2 次，B 命中 1 次
    a = [c for c in chunks if c.metadata.get("dish_name") == "a"]
    b = [c for c in chunks if c.metadata.get("dish_name") == "b"]
    selected = [a[0], a[1], b[0]]

    parents = m.get_parent_documents(selected)
    assert len(parents) == 2
    assert parents[0].metadata["dish_name"] == "a"
    assert parents[1].metadata["dish_name"] == "b"


def test_statistics_and_export(tmp_path: Path):
    write_file(tmp_path / "drink" / "奶茶.md", "# 奶茶\n★")
    m = DataPreparationModule(str(tmp_path))
    m.load_documents()
    m.chunk_documents()

    stats = m.get_statistics()
    assert isinstance(stats, dict)
    assert stats["total_documents"] == 1
    assert stats["total_chunks"] >= 1
    assert "categories" in stats
    assert "difficulties" in stats

    out = tmp_path / "meta.json"
    m.export_metadata(str(out))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["dish_name"] == "奶茶"
