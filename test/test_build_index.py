import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _fake_doc(text, meta=None):
    return SimpleNamespace(page_content=text, metadata=meta or {})


def _load_module():
    # Lightweight stubs let tests run without installing heavy dependencies.
    if "langchain_huggingface" not in sys.modules:
        fake_hf = ModuleType("langchain_huggingface")
        fake_hf.HuggingFaceEmbeddings = object
        sys.modules["langchain_huggingface"] = fake_hf

    if "langchain_community" not in sys.modules:
        sys.modules["langchain_community"] = ModuleType("langchain_community")
    if "langchain_community.vectorstores" not in sys.modules:
        fake_vs = ModuleType("langchain_community.vectorstores")
        fake_vs.FAISS = type("FAISS", (), {})
        sys.modules["langchain_community.vectorstores"] = fake_vs

    if "langchain_core" not in sys.modules:
        sys.modules["langchain_core"] = ModuleType("langchain_core")
    if "langchain_core.documents" not in sys.modules:
        fake_docs = ModuleType("langchain_core.documents")
        fake_docs.Document = object
        sys.modules["langchain_core.documents"] = fake_docs

    if "torch" not in sys.modules:
        fake_torch = ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        sys.modules["torch"] = fake_torch

    module_path = Path(__file__).resolve().parents[1] / "module" / "build_index.py"
    spec = importlib.util.spec_from_file_location("build_index_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


@pytest.fixture
def work_tmp():
    p = Path(__file__).resolve().parent / ".tmp_test_build_index"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stub_embedding_constructor(monkeypatch, mod):
    calls = []

    def fake_embeddings(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(name="fake_embeddings")

    monkeypatch.setattr(mod, "HuggingFaceEmbeddings", fake_embeddings)
    return calls


def _build_module(monkeypatch, mod, work_tmp):
    _stub_embedding_constructor(monkeypatch, mod)
    return mod.IndexBuildModule(
        model_name="fake-model",
        index_save_path=str(work_tmp / "vector_index"),
    )


def test_setup_embeddings_init_cpu_and_normalize(monkeypatch, mod, work_tmp):
    calls = _stub_embedding_constructor(monkeypatch, mod)
    monkeypatch.setattr(mod.torch.cuda, "is_available", lambda: False)

    obj = mod.IndexBuildModule(
        model_name="my-model",
        index_save_path=str(work_tmp / "index"),
    )

    assert obj.embeddings is not None
    assert len(calls) == 1
    kwargs = calls[0]["kwargs"]
    assert kwargs["model_name"] == "my-model"
    assert kwargs["model_kwargs"] == {"device": "cpu"}
    assert kwargs["encode_kwargs"] == {"normalize_embeddings": True}


def test_build_vector_index_raises_on_empty_chunks(monkeypatch, mod, work_tmp):
    obj = _build_module(monkeypatch, mod, work_tmp)

    with pytest.raises(ValueError):
        obj.build_vector_index([])


def test_build_vector_index_calls_faiss_from_texts(monkeypatch, mod, work_tmp):
    obj = _build_module(monkeypatch, mod, work_tmp)
    docs = [_fake_doc("a", {"x": 1}), _fake_doc("b", {"x": 2})]
    fake_store = SimpleNamespace()
    called = {}

    def fake_from_texts(*, texts, embedding, metadatas):
        called["texts"] = texts
        called["embedding"] = embedding
        called["metadatas"] = metadatas
        return fake_store

    monkeypatch.setattr(mod.FAISS, "from_texts", fake_from_texts, raising=False)

    result = obj.build_vector_index(docs)

    assert result is fake_store
    assert obj.vectorstore is fake_store
    assert called["texts"] == ["a", "b"]
    assert called["metadatas"] == [{"x": 1}, {"x": 2}]
    assert called["embedding"] is obj.embeddings


def test_save_index_raises_if_vectorstore_missing(monkeypatch, mod, work_tmp):
    obj = _build_module(monkeypatch, mod, work_tmp)

    with pytest.raises(ValueError):
        obj.save_index()


def test_save_index_creates_dir_and_saves(monkeypatch, mod, work_tmp):
    obj = _build_module(monkeypatch, mod, work_tmp)
    target_dir = work_tmp / "nested" / "index"
    obj.index_save_path = str(target_dir)
    calls = {}

    class FakeStore:
        def save_local(self, path):
            calls["path"] = path

    obj.vectorstore = FakeStore()
    obj.save_index()

    assert target_dir.exists()
    assert calls["path"] == str(target_dir)


def test_load_index_returns_none_when_path_not_exists(monkeypatch, mod, work_tmp):
    obj = _build_module(monkeypatch, mod, work_tmp)
    obj.index_save_path = str(work_tmp / "missing_index_dir")

    result = obj.load_index()
    assert result is None


def test_load_index_calls_faiss_load_local(monkeypatch, mod, work_tmp):
    obj = _build_module(monkeypatch, mod, work_tmp)
    index_dir = work_tmp / "exists_index_dir"
    index_dir.mkdir(parents=True, exist_ok=True)
    obj.index_save_path = str(index_dir)
    fake_store = SimpleNamespace()
    called = {}

    def fake_load_local(path, embeddings, allow_dangerous_deserialization):
        called["path"] = path
        called["embeddings"] = embeddings
        called["allow"] = allow_dangerous_deserialization
        return fake_store

    monkeypatch.setattr(mod.FAISS, "load_local", fake_load_local, raising=False)

    result = obj.load_index()

    assert result is fake_store
    assert obj.vectorstore is fake_store
    assert called["path"] == str(index_dir)
    assert called["embeddings"] is obj.embeddings
    assert called["allow"] is True


def test_add_documents_requires_vectorstore(monkeypatch, mod, work_tmp):
    obj = _build_module(monkeypatch, mod, work_tmp)
    with pytest.raises(ValueError):
        obj.add_documents([_fake_doc("new", {})])


def test_add_documents_calls_vectorstore(monkeypatch, mod, work_tmp):
    obj = _build_module(monkeypatch, mod, work_tmp)
    docs = [_fake_doc("new", {"a": 1})]
    calls = {}

    class FakeStore:
        def add_documents(self, new_chunks):
            calls["new_chunks"] = new_chunks

    monkeypatch.setattr(
        mod,
        "logger",
        SimpleNamespace(info=lambda *args, **kwargs: None),
        raising=False,
    )
    obj.vectorstore = FakeStore()
    obj.add_documents(docs)
    assert calls["new_chunks"] == docs


def test_similarity_search_requires_vectorstore(monkeypatch, mod, work_tmp):
    obj = _build_module(monkeypatch, mod, work_tmp)
    with pytest.raises(ValueError):
        obj.similarity_search("query", k=3)


def test_similarity_search_calls_vectorstore(monkeypatch, mod, work_tmp):
    obj = _build_module(monkeypatch, mod, work_tmp)
    expected_docs = [_fake_doc("hit", {"score": 1})]
    calls = {}

    class FakeStore:
        def similarity_search(self, query, k=5):
            calls["query"] = query
            calls["k"] = k
            return expected_docs

    obj.vectorstore = FakeStore()
    result = obj.similarity_search("宫保鸡丁", k=2)

    assert calls == {"query": "宫保鸡丁", "k": 2}
    assert result == expected_docs
