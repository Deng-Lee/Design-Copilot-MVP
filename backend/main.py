import os
import sys

# 1. 强制设置镜像 (必须加，否则加载 BAAI 模型可能会联网报错)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    print("❌ 错误: 未找到 langchain-classic 包。")
    print("👉 请运行: pip install langchain-classic")
    sys.exit(1)

# 加载环境变量 (读取 API Key)
load_dotenv("../.env")

# ================= 配置区域 =================

# 1. 数据库路径 (必须与 ingest.py 里的 PERSIST_DIRECTORY 完全一致)
current_dir = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(current_dir, "chroma_db")

# 2. Embedding 模型 (必须与 ingest.py 里的模型完全一致)
# 你刚才用的是这个中文模型，这里读取时必须用同一个
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

# 3. 大模型配置 (这里默认使用 DeepSeek API，效果最好)
# 如果你想用本地 Ollama，请看代码底部的注释进行修改
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or "你的sk-xxxxxxxx" 

# ===========================================

def main():
    # --- 1. 准备“钥匙” (Embedding) ---
    print(f"🔑 正在加载 Embedding 模型: {EMBEDDING_MODEL_NAME}...")
    try:
        embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("💡 提示: 如果是网络问题，请检查 HF_ENDPOINT 设置或尝试手动下载模型。")
        return

    # --- 2. 打开“仓库” (ChromaDB) ---
    if not os.path.exists(PERSIST_DIRECTORY):
        print(f"❌ 错误: 找不到数据库文件夹 {PERSIST_DIRECTORY}")
        print("👉 请先运行 ingest.py 生成数据！")
        return

    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    print(f"📚 成功连接数据库，当前包含 {vector_db._collection.count()} 条知识片段")

    # --- 3. 唤醒“大脑” (LLM) ---
    print("🤖 正在连接 DeepSeek 大模型...")
    llm = ChatOpenAI(
        model="deepseek-chat", 
        api_key=DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com",
        temperature=0.1 # 写代码需要严谨，温度设低一点
    )

    # --- 4. 设定“指令” (Prompt) ---
    # 这是 RAG 的核心：告诉模型“参考下面的 Context 来回答 Question”
    prompt = ChatPromptTemplate.from_template("""
    你是一个资深的前端开发专家 (Design Copilot)。
    请根据以下 <context> 标签中的参考文档，回答用户的 <input>。
    
    <context>
    {context}
    </context>
    
    <input>
    {input}
    </input>

    【回答要求】：
    1. 必须优先使用参考文档中提供的组件 API 和代码风格。
    2. 如果参考文档有相关代码，直接给出完整的、可运行的代码示例。
    3. 如果文档里没提到的属性，不要瞎编。
    """)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # 步骤 B: 创建“检索链”
    # 它的作用是：拿到用户问题 -> 调用 retriever -> 拿到相关文档 -> 传给上面的 question_answer_chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # --- 6. 交互循环 ---
    print("\n✅ Design Copilot 已就绪！(输入 'exit' 退出)")
    while True:
        user_input = input("\n👉 请输入需求: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        
        if not user_input.strip():
            continue

        print("⏳ 思考中...")
        try:
            # 执行问答链
            response = rag_chain.invoke({"input": user_input})
            
            print("\n" + "="*40)
            print("🤖 Copilot 回答：")
            print(response["answer"])
            print("="*40)
            
            # 调试：看看它到底参考了哪里
            print("\n📚 参考来源：")
            for doc in response["context"]:
                # 获取我们在 ingest.py 里辛苦保存的文件名 metadata
                source = doc.metadata.get('source', '未知来源')
                filename = os.path.basename(source)
                print(f"- {filename}")
                
        except Exception as e:
            print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    main()