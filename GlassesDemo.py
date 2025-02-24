import re
import os
import h5py
import logging
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = 'paraphrase-multilingual-mpnet-base-v2'


# ================== 1. 向量化类（含 Faiss 接口）==================
class Vectorizer:
    def __init__(self, model_name='paraphrase-multilingual-mpnet-base-v2'):
        # 自动选择设备
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        device = "cpu"  # Faiss可能只支持CPU

        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        logger.info(f"✅ 模型加载完成: {model_name}（设备: {self.device}）")

    def batch_encode(self, texts, batch_size=64, normalize=True):
        """批量文本向量化"""
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = self.model.encode(batch,
                                    convert_to_numpy=True,
                                    normalize_embeddings=normalize,
                                    show_progress_bar=True)
            embeddings.append(emb)
        return np.vstack(embeddings)


# ================== 2. Faiss 接口封装（可替换其他引擎）==================
class FaissIndexer:
    @staticmethod
    def build(vectors):
        """构建索引（通用接口）"""
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)  # 使用内积（余弦相似度）
        faiss.normalize_L2(vectors)  # 归一化处理
        index.add(vectors)
        return index

    @staticmethod
    def save(index, file_path):
        """保存索引"""
        faiss.write_index(index, file_path)

    @staticmethod
    def load(file_path):
        """加载索引"""
        return faiss.read_index(file_path)


# ================== 3. 数据处理（保持原有功能）==================
def clean_text(text):
    """清理文本"""
    if isinstance(text, str):
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()
    return ""


def read_file():
    """读取 CSV 文件"""
    df = pd.read_csv("testTrainData.csv",
                     delimiter=",",
                     encoding="utf-8-sig",
                     index_col=0)
    df.columns = df.columns.str.strip()

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    return df


# ================== 4. HDF5 缓存（集成索引存储）==================
HDF5_FILENAME = f"processed_data_{model_name}.h5"


def save_cache(df, vectors):
    """保存数据和索引"""
    with h5py.File(HDF5_FILENAME, "w") as f:
        # 存储向量
        f.create_dataset("embeddings", data=vectors)

        # 存储元数据
        metadata_group = f.create_group("metadata")
        for col in df.columns:
            str_dtype = h5py.string_dtype(encoding='utf-8')
            data = df[col].astype(str).values.astype(str_dtype)
            metadata_group.create_dataset(col, data=data)

        # 存储 Faiss 索引
        index = FaissIndexer.build(vectors)
        FaissIndexer.save(index, "temp.index")
        with open("temp.index", "rb") as f_index:
            f.create_dataset("faiss_index", data=np.void(f_index.read()))

        # 记录时间戳
        f.attrs["create_time"] = datetime.now().isoformat()

    os.remove("temp.index")
    logger.info(f"✅ 数据+索引已缓存到 {HDF5_FILENAME}")


def load_cache():
    """加载缓存（返回数据+索引）"""
    if os.path.exists(HDF5_FILENAME):
        with h5py.File(HDF5_FILENAME, "r") as f:
            logger.info("✅ 加载缓存数据...")

            # 加载数据
            df = pd.DataFrame({col: list(f["metadata"][col])
                               for col in f["metadata"].keys()})
            vectors = np.array(f["embeddings"])

            # 加载索引
            index_data = bytes(f["faiss_index"][()])
            with open("temp.index", "wb") as f_temp:
                f_temp.write(index_data)
            index = FaissIndexer.load("temp.index")
            os.remove("temp.index")

            return df, vectors, index
    return None, None, None


# ================== 5. 主处理流程（返回索引）==================
def process_data(clean_cache=False):
    """主处理流程（返回索引）"""
    if clean_cache:
        if os.path.exists(HDF5_FILENAME):
            os.remove(HDF5_FILENAME)
        logger.info("🗑️ 缓存已清除，重新计算...")

    # 尝试加载缓存
    cached_df, cached_vectors, cached_index = load_cache()
    if cached_df is not None:
        logger.info("✅ 成功加载缓存")
        return cached_df, cached_vectors, cached_index

    # 处理新数据
    df = read_file()
    text_cols = [col for col in df.columns if df[col].dtype == "object"]

    for col in text_cols:
        df[col] = df[col].fillna("").apply(clean_text)

    df["text"] = df[text_cols].apply(lambda x: " ".join(x), axis=1)

    vectorizer = Vectorizer()
    vectors = vectorizer.batch_encode(df["text"].tolist())

    save_cache(df, vectors)
    _, _, index = load_cache()  # 确保索引加载

    return df, vectors, index


# ================== 6. 语义搜索接口 ==================
class QueryRequest(BaseModel):
    queries: List[str]

app = FastAPI()

@app.post("/search/")
async def search(query: QueryRequest):
    df, vectors, index = load_cache()

    # 将查询转化为向量
    vectorizer = Vectorizer()
    query_vectors = vectorizer.batch_encode(query.queries)
    faiss.normalize_L2(query_vectors)

    # 执行搜索
    scores, indices = index.search(query_vectors, top_k=5)

    results = []
    for q_scores, q_indices in zip(scores, indices):
        result = []
        for score, idx in zip(q_scores, q_indices):
            result.append({"score": float(score), "text": str(df.iloc[idx]["text"])})
        results.append(result)

    return {"results": results}


# ================== 7. 启动 FastAPI 应用 ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)