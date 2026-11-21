import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from ConfigSpace import Configuration

class BOPROEmbedding:
    """
    BOPRO 专用组件 (Stateful Singleton)：
    1. 维护 SimCSE 模型。
    2. 维护历史配置的 Embedding 缓存，避免重复计算。
    """
    _instance = None
    
    def __init__(self):
        print("[BOPRO] Loading SimCSE model... (this may take a while for the first time)")
        # 你的模型路径
        self.model = SentenceTransformer('/root/RAG/models--sentence-transformers--all-MiniLM-L6-v2')
        
        # [新增] 缓存列表，存储 numpy array (1, 768)
        self.cached_embeddings = [] 
        
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def reset(self):
        """清空缓存，用于新的实验开始时"""
        self.cached_embeddings = []

    def config_to_text(self, config: Configuration) -> str:
        """
        将配置字典序列化为 BOPRO 风格的文本。
        """
        if isinstance(config, Configuration):
            cfg_dict = config.get_dictionary()
        else:
            cfg_dict = config 
            
        items = sorted(cfg_dict.items())
        text = ", ".join([f"{k}: {v}" for k, v in items if v is not None])
        return text

    def add_observation(self, config: Configuration):
        """
        [关键优化] 增量添加一个配置到缓存中。
        只在评估完成后调用一次，避免重复编码。
        """
        text = self.config_to_text(config)
        # encode 返回 (768,) -> reshape 为 (1, 768) 方便后续堆叠
        embedding = self.model.encode([text], convert_to_numpy=True)[0].reshape(1, -1)
        self.cached_embeddings.append(embedding)

    def get_similarity_scores(self, proposal_config: Configuration) -> np.ndarray:
        """
        计算 proposal_config 与 缓存历史 的余弦相似度。
        """
        if not self.cached_embeddings:
            return np.array([])

        # 1. 只编码当前的 Proposal (1次 forward pass)
        prop_text = self.config_to_text(proposal_config)
        prop_emb = self.model.encode([prop_text], convert_to_numpy=True)[0].reshape(1, -1)
        
        # 2. 堆叠历史缓存 (快速)
        # 将 list of arrays 转换为 (N, 768) 矩阵
        history_matrix = np.vstack(self.cached_embeddings)
        
        # 3. 计算相似度
        # cosine_similarity 返回 (1, N)，取 [0] 变成 (N,)
        sim_scores = cosine_similarity(prop_emb, history_matrix)[0]
        
        return sim_scores