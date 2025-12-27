import os
import sys
import streamlit as st

# 1. 消除 Tokenizers 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 2. 强制国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 适配 LangChain v1.2 的引用
try:
    from langchain_classic.chains import create_retrieval_chain
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    print("❌ 错误: 未找到 langchain-classic 包。")
    print("👉 请运行: pip install langchain-classic")
    sys.exit(1)

# 加载环境变量
load_dotenv("../.env")

# ================= 配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIRECTORY = os.path.join(current_dir, "chroma_db")
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# ================= 页面设置 =================
st.set_page_config(page_title="Design Copilot", page_icon="🤖", layout="wide")
st.title("🤖 Design Copilot (RAG)")

# ================= 核心逻辑 (带缓存) =================
@st.cache_resource
def load_chain():
    """
    加载模型和数据库。
    使用 cache_resource 装饰器，确保只加载一次，
    防止每次发消息都重新加载模型。
    """
    print("🔄 正在初始化 RAG 链...")
    
    # 1. 加载 Embedding
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # 2. 连接 Chroma
    if not os.path.exists(PERSIST_DIRECTORY):
        st.error(f"找不到数据库: {PERSIST_DIRECTORY}")
        return None
        
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    # 3. 加载 LLM
    llm = ChatOpenAI(
        model="deepseek-chat", 
        api_key=DEEPSEEK_API_KEY, 
        base_url="https://api.deepseek.com",
        temperature=0.1
    )
    
    # 4. Prompt
    prompt = ChatPromptTemplate.from_template("""
    你是一个资深的前端开发专家 (Design Copilot)。
    请根据以下 <context> 标签中的参考文档，回答用户的 <input>。
    
    <context>
    {context}
    </context>
    
    <input>
    {input}
    </input>

    【要求】：
    1. 优先使用参考文档中的组件 API。
    2. 直接给出完整的、可运行的代码。
    3. 如果文档未提及，请说明。
    """)
    
    # 5. 构建链
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("✅ RAG 链初始化完成")
    return rag_chain

# 加载链 (只会运行一次)
chain = load_chain()

# ================= 聊天界面逻辑 =================

# 1. 初始化聊天历史 (Session State)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. 处理用户输入
if prompt := st.chat_input("请输入你的需求 (例如: 给我一个带图标的按钮)"):
    # 显示用户消息
    with st.chat_message("user"):
        st.markdown(prompt)
    # 记录到历史
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 生成回答
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("⏳ 正在思考并检索文档...")
        
        try:
            # 调用 RAG 链
            response = chain.invoke({"input": prompt})
            answer = response["answer"]
            
            # 格式化一下参考来源 (可选)
            sources_text = "\n\n**📚 参考文档：**\n"
            seen_sources = set()
            for doc in response["context"]:
                source_name = os.path.basename(doc.metadata.get('source', '未知'))
                if source_name not in seen_sources:
                    sources_text += f"- `{source_name}`\n"
                    seen_sources.add(source_name)
            
            # 显示最终结果
            full_response = answer + sources_text
            message_placeholder.markdown(full_response)
            
            # 记录助手回复
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            message_placeholder.error(f"❌ 发生错误: {e}")
            