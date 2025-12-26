import os
# 🔥 必须加镜像，否则加载模型时会联网验证然后报错
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 🔥 确保路径和 ingest.py 一模一样
current_dir = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(current_dir, "chroma_db")

def check_db():
    print(f"🕵️ 正在检查数据库: {PERSIST_DIRECTORY}")
    
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding)
    
    count = db._collection.count()
    print(f"📊 数据库当前包含片段数: {count}")
    
    if count > 0:
        print("✅ 验证成功！数据库非空。")
        # 简单测试一下搜索
        results = db.similarity_search("button", k=1)
        if results:
            print(f"📝 试搜索 'button' 结果: {results[0].page_content[:50]}...")
    else:
        print("❌ 依然是空的，请检查 ingest.py 是否报错。")

if __name__ == "__main__":
    check_db()