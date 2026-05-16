为了构建李白“一生中遇到的当官的诗人”信息查询系统，我们将使用 nano-graphrag 来实现。这完美契合了GraphRAG的核心优势：它不仅能检索到“李白”的片段，还能通过知识图谱挖掘出“李白-朋友-杜甫-官职”这样的多跳复杂关系。

下面我将为你构建一个完整的简单脚本，包含Mock数据（李白生平轨迹片段）和基于 Ollama 本地大模型的检索代码。

📂 第一步：准备 Mock 数据（李白生平轨迹）

我们将李白的生平拆分为几个带有丰富实体关系的文本片段，保存为 libai_life.txt。这些数据特意埋藏了需要“多跳推理”才能找到的线索。

公元744年，李白在洛阳遇到了比他小11岁的杜甫。当时的杜甫虽然诗名渐盛，但仕途坎坷，仅在朝廷担任过短暂的“左拾遗”一职，这是一个从八品上的谏官。两人一见如故，结下了深厚的友谊，随后一同游历梁宋。

公元742年，李白奉诏入京，担任翰林供奉。在长安期间，他结识了时任太子宾客的贺知章。贺知章不仅是著名的诗人，更是朝廷重臣，官至正三品。贺知章读到李白的《蜀道难》后，惊叹地称他为“谪仙人”，并解下腰间金龟换酒，与李白痛饮。

公元757年，李白因卷入永王李璘案被流放夜郎。在流放途中，他遇到了同样仕途失意的好友高适。高适是唐代著名边塞诗人，但他也是一位杰出的官员，曾官至剑南节度使，封渤海县侯，是唐代诗人中政治地位极高的一位。

公元744年秋，李白与杜甫、高适三位大诗人曾在梁宋（今河南商丘）一带共同游猎。当时的高适尚未发迹，而杜甫也只是担任左拾遗这样的小官。李白则在不久前刚刚被“赐金放还”，结束了他在长安翰林院的供奉生涯。

💻 第二步：构建 nano-graphrag 检索脚本

由于 nano-graphrag 依赖大语言模型（LLM）来提取实体和关系，为了保证你无需配置复杂的商业 API Key，下面的脚本适配了 Ollama（本地大模型）。

在运行前，请确保你已安装 Ollama 并拉取了基础模型（如 ollama pull qwen2.5 或 ollama pull llama3）。

import os
import asyncio
import ollama
from nano_graphrag import GraphRAG, QueryParam

配置本地 Ollama 模型
OLLAMA_LLM_MODEL = "qwen2.5"  # 或者使用 "llama3"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

定义适配 Ollama 的 LLM 函数
async def ollama_model_if_cache(prompt, system_prompt=None, history_messages=[], **kwargs):
    # 移除 Ollama 不支持的参数
    kwargs.pop("max_tokens", None)
    kwargs.pop("response_format", None)
    
    ollama_client = ollama.AsyncClient()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    
    response = await ollama_client.chat(model=OLLAMA_LLM_MODEL, messages=messages, **kwargs)
    return response.message.content

定义适配 Ollama 的 Embedding 函数
async def ollama_embedding(texts: list[str]) -> list[list[float]]:
    # 如果是单个字符串，转为列表
    if isinstance(texts, str):
        texts = [texts]
    response = ollama.embeddings(model=OLLAMA_EMBEDDING_MODEL, input=texts)
    return [r["embedding"] for r in response["embeddings"]]

async def main():
    WORKING_DIR = "./libai_graphrag_db"
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)

    # 初始化 GraphRAG
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_if_cache,
        cheap_model_func=ollama_model_if_cache,
        embedding_func=ollama_embedding
    )

    # 读取 Mock 数据并插入知识库
    print("📚 正在读取李白生平文献并构建知识图谱...")
    with open("./libai_life.txt", encoding="utf-8") as f:
        libai_text = f.read()
    
    # 插入数据，nano-graphrag 会自动提取实体（李白、杜甫、高适...）和关系（朋友、官职...）
    await rag.ainsert(libai_text)
    print("✅ 知识图谱构建完成！n")

    # 3. 提出复杂查询：李白一生中遇到的当官的诗人都有谁？
    query = "李白一生中遇到的当官的诗人都有哪些人？请列出他们的名字以及担任的官职。"
    print(f"🔍 正在提问：{query}")
    
    # 使用 global 模式，让 GraphRAG 结合整个知识图谱进行综合推理
    result = await rag.aquery(query, param=QueryParam(mode="global"))
    
    print("n📜 GraphRAG 回答：")
    print(result)

if name == "main":
    # 运行异步主函数
    asyncio.run(main())

💡 为什么这个场景必须用 GraphRAG？

多跳推理（Multi-hop Reasoning）：
   在 Mock 数据中，关于“高适”的信息是分散的。一段提到了“李白遇到高适”，另一段提到了“高适官至剑南节度使”。
   普通 RAG：可能会检索到“李白遇到高适”的片段，但如果没有精准匹配到“高适是官员”的片段，就会漏掉答案；或者检索到了高适的官职，却无法将其与“李白的朋友”这个条件强关联。
   GraphRAG：会在图谱中建立 李白 --(遇到)--> 高适 和 高适 --(担任)--> 剑南节度使 的连线。当你提问时，它能顺着图谱路径直接找到答案。

全局视角（Global Understanding）：
   你的问题是“一生中遇到的...”，这要求模型不能只看某一个文本块，而是要遍历李白整个社交图谱。QueryParam(mode="global") 会利用 GraphRAG 的社区摘要功能，把李白在不同时期（洛阳、长安、梁宋）遇到的所有人物汇总起来进行筛选。

🛠️ 运行前置准备
安装依赖：pip install nano-graphrag ollama
确保本地 Ollama 服务已开启，并拉取了模型：
   ollama pull qwen2.5 (推荐中文能力强的模型)
   ollama pull nomic-embed-text (用于文本向量化)
将 Mock 数据保存为 libai_life.txt，与脚本放在同一目录下运行即可。