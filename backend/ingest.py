import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(current_dir, "chroma_db")

# 1. 加载环境变量
load_dotenv("../.env") # 读取上一级目录的 .env

# 检查 Key 是否存在
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ 错误: 未找到 GOOGLE_API_KEY，请检查 .env 文件")
    exit()

# 数据库存储路径

def main():
    print(f"📂 数据库将存放在: {PERSIST_DIRECTORY}")

    # 2. 加载数据：扫描 data 目录下的所有 .md 文件
    # glob="*.md" 表示只看 markdown 文件
    loader = DirectoryLoader('./data', glob="*.md", loader_cls=TextLoader)
    documents = loader.load()
    print(f"📄 加载了 {len(documents)} 个文档")

    # 3. 文本切片 (Chunking)
    # 为什么是 1000？因为组件文档包含表格和长代码，切太小会断章取义
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, # 重叠部分，防止切断关键上下文
        separators=["\n## ", "\n### ", "\n", " ", ""] # 优先按标题切分
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✂️ 切分成了 {len(chunks)} 个片段")

    # 4. 向量化并存储 (Embedding & Storage)
    print("💾 正在存入 ChromaDB (这可能需要几秒钟)...")
    
    vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), # 使用 HuggingFace 模型
            persist_directory=PERSIST_DIRECTORY
        )
    
    
    # 自动下载模型、计算向量、存入本地文件夹
    
    print(f"✅ 成功！数据库已保存在 {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    main()