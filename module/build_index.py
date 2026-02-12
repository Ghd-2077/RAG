import logging
from typing import List
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import torch
class IndexBuildModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5",
                 index_save_path: str = "./vector_index"):
        self.model_name = model_name
        self.index_save_path = index_save_path#本地的向量缓存路径
        self.embeddings = None#嵌入模型实例
        self.vectorstore = None #向量数据库
        self.setup_embeddings()
    #实例化嵌入模型
    def setup_embeddings(self):
        #根据机器情况选择模型运行的设备
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
        model_name=self.model_name,#指定要加载的 HuggingFace 嵌入模型名
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    #根据切分后的chunks构建索引
    #是否可以和下面的加载缓存代码合并，来去除冗余代码并增强可读性
    def build_vector_index(self, chunks: List[Document]) -> FAISS:
    
        if not chunks:
            raise ValueError("文档块列表不能为空")
        
        # 提取chunks中的文本内容和元数据
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # 构建FAISS向量索引，返回一个向量数据库实例
        self.vectorstore = FAISS.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas
        )
        #FAISS构建索引的流程如下：
        """
        首先将一批chunk转为嵌入向量，然后组织成矩阵[chunk数，embeding维度]
        将这个矩阵存入FAISS的索引对象
        为原始文档生成对应的id，和索引对应
        """
        
        return self.vectorstore
    #索引缓存
    def save_index(self):
        """保存向量索引到配置的路径"""
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")
        
        # 确保保存目录存在
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        
        self.vectorstore.save_local(self.index_save_path)
    #缓存加载
    def load_index(self):
        
        #检查缓存路径是否存在
        if not Path(self.index_save_path).exists():
            return None
        #实例化向量嵌入模型，在定义实例的时候已经做了初始化模型函数，这里不用在写了吧？
        """ if not self.embeddings:
            self.setup_embeddings() """
        #加载本地数据库，返回数据库实例
        self.vectorstore = FAISS.load_local(
            self.index_save_path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        return self.vectorstore

    #上述操作完成了数据库的构建，接下来进行数据库增删改查函数的实现
    ##根据需要，暂时仅实现增和查功能
    def add_documents(self, new_chunks: List[Document]):
        """
        向现有索引添加新文档
        
        Args:
            new_chunks: 新的文档块列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建向量数据库")
        
        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        self.vectorstore.add_documents(new_chunks)
        logger.info("新文档添加完成")
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        相似度搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            相似文档列表
        """
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量数据库")
        
        return self.vectorstore.similarity_search(query, k=k)