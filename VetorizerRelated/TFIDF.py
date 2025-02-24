

'''  简单使用TF_IDF demo '''

# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# # 🔹 1. 创建文本数据
# documents = [
#     "苹果 是 一种 水果",
#     "苹果 和 香蕉 是 我 喜欢 吃 的 水果",
#     "我 讨厌 吃 香蕉 但是 喜欢 吃 苹果",
#     "水果 里面 我 最 喜欢 苹果"
# ]
#
# # 🔹 2. 初始化 TF-IDF 向量器
# vectorizer = TfidfVectorizer()
#
# # 🔹 3. 计算 TF-IDF 矩阵
# tfidf_matrix = vectorizer.fit_transform(documents)
#
# # 🔹 4. 获取单词列表
# feature_names = vectorizer.get_feature_names_out()
#
# # 🔹 5. 转换为 DataFrame 方便查看
# df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
#
# # 🔹 6. 打印 TF-IDF 结果
# print(df_tfidf)


'''Word2Vec 进阶版，明确语义相似度'''
from gensim.models import Word2Vec

# 示例语料库
sentences = [
    ["苹果", "是", "一种", "水果"],
    ["苹果", "和", "香蕉", "是", "我", "喜欢", "吃", "的", "水果"],
    ["我", "讨厌", "吃", "香蕉", "但是", "喜欢", "吃", "苹果"],
    ["水果", "里面", "我", "最", "喜欢", "苹果"]
]

# 训练 Word2Vec 模型
model = Word2Vec(sentences, vector_size=50, window=5, min_count=1, workers=4)

# 查看 "苹果" 的词向量
print(model.wv["苹果"])

# 找出最相似的词
print(model.wv.most_similar("苹果"))