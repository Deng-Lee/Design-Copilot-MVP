import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(current_dir, "chroma_db")

# 1. 加载环境变量
load_dotenv("../.env") # 读取上一级目录的 .env

# 检查 Key 是否存在
if not os.getenv("DEEPSEEK_API_KEY"):
    print("❌ 错误: 未找到 DEEPSEEK_API_KEY，请检查 .env 文件")
    exit()

# 数据库存储路径

def main():
    print(f"📂 数据库将存放在: {PERSIST_DIRECTORY}")

    # 2. 加载数据：扫描 data 目录下的所有 .md 文件
    # glob="*.md" 表示只看 markdown 文件
    loader = DirectoryLoader('./data', glob="*.md", loader_cls=TextLoader)
    documents = loader.load()
    print(f"📄 加载了 {len(documents)} 个文档")
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    md_header_splits = []
    
    for doc in documents:
        # 对每个文档的内容进行标题切分
        splits = markdown_splitter.split_text(doc.page_content)
        
        # 【关键步骤】MarkdownSplitter 切完后会丢失原来的 file_path (source)
        # 我们必须手动把原文档的 metadata (比如文件名) 更新到新切片里
        for split in splits:
            split.metadata.update(doc.metadata)
            
        md_header_splits.extend(splits)

    print(f"🧩 按标题切分后得到了 {len(md_header_splits)} 个语义片段")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""] # 二次切分就不需要再关注标题了，主要关注段落
    )
    
    final_chunks = text_splitter.split_documents(md_header_splits)
    print(f"✂️ 最终切分成了 {len(final_chunks)} 个片段")

    # 4. 向量化并存储 (Embedding & Storage)
    print("💾 正在存入 ChromaDB (这可能需要几秒钟)...")
    
    vector_store = Chroma.from_documents(
            documents=final_chunks,
            embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5"), # 使用 HuggingFace 模型
            persist_directory=PERSIST_DIRECTORY
        )
    
    
    # 自动下载模型、计算向量、存入本地文件夹
    
    print(f"✅ 成功！数据库已保存在 {PERSIST_DIRECTORY}")

if __name__ == "__main__":
    main()