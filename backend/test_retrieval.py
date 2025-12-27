import os
# 1. 如果之前因为网络问题加了这行，这里也要加，否则加载模型会卡住
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(current_dir, "chroma_db")

def test_search():
    # 2. 必须使用和 ingest.py 中完全一样的 Embedding 模型
    print("正在加载 Embedding 模型...")
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    # 3. 加载已经存在的数据库
    # 注意：persist_directory 必须和你 ingest.py 里写的路径完全一致
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)

    # 4. 模拟一个用户问题
    query = "设置按钮状态" 
    
    print(f"正在搜索问题: {query}")
    # 5. 搜索相似度最高的 3 个片段
    docs = db.similarity_search(query, k=3)

    print(f"\n找到 {len(docs)} 个相关片段：\n")
    for i, doc in enumerate(docs):
        print(f"--- 片段 {i+1} ---")
        print(doc.page_content[:200] + "...") # 只打印前200个字预览
        print("----------------\n")

if __name__ == "__main__":
    test_search()