#!/usr/bin/env python3
"""
李白'一生中遇到的当官的诗人'信息查询系统
基于 nano-graphrag 和 Ollama 本地大模型
"""

import os
import asyncio
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam

# 配置本地 Ollama 模型
OLLAMA_LLM_MODEL = "phi4-mini:3.8b"  # 使用较小的模型进行测试
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# 定义适配 Ollama 的 LLM 函数
async def ollama_model_func(prompt: str, **kwargs) -> str:
    """异步Ollama模型完成函数"""
    response = ollama.generate(
        model=OLLAMA_LLM_MODEL,
        prompt=prompt,
        options={'temperature': 0.1, 'num_predict': 2048}
    )
    result = response['response']
    if '<|COMPLETE|>' not in result:
        result = result.strip() + '\n<|COMPLETE|>'
    return result

# 定义适配 Ollama 的 Embedding 函数
async def ollama_embedding_func(text, **kwargs) -> np.ndarray:
    """异步Ollama嵌入函数"""
    if isinstance(text, list):
        text = text[0]
    
    response = ollama.embeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        prompt=text
    )
    return np.array(response['embedding'], dtype=np.float32)

# 设置嵌入维度
ollama_embedding_func.embedding_dim = 768

async def main():
    """主函数"""
    print("=" * 70)
    print("📚 李白'一生中遇到的当官的诗人'信息查询系统")
    print("基于 nano-graphrag 和 Ollama 本地大模型")
    print("=" * 70)
    
    WORKING_DIR = "./libai_graphrag_db"
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    
    # 初始化 GraphRAG
    print("\n🔧 正在初始化 GraphRAG...")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_func,
        cheap_model_func=ollama_model_func,
        embedding_func=ollama_embedding_func,
        enable_local=True,
        embedding_batch_num=1,
    )
    print("✅ GraphRAG 初始化完成！")
    
    # 读取 Mock 数据并插入知识库
    print("\n📖 正在读取李白生平文献并构建知识图谱...")
    with open("./libai_life.txt", encoding="utf-8") as f:
        libai_text = f.read()
    
    print(f"📄 已读取 {len(libai_text)} 个字符的文献")
    
    # 插入数据，nano-graphrag 会自动提取实体（李白、杜甫、高适...）和关系（朋友、官职...）
    print("\n🔄 正在构建知识图谱（这可能需要几分钟）...")
    await rag.ainsert(libai_text)
    print("✅ 知识图谱构建完成！\n")
    
    # 提出复杂查询：李白一生中遇到的当官的诗人都有谁？
    query = "李白一生中遇到的当官的诗人都有哪些人？请列出他们的名字以及担任的官职。"
    print("=" * 70)
    print(f"🔍 正在提问：{query}")
    print("=" * 70)
    
    # 使用 global 模式，让 GraphRAG 结合整个知识图谱进行综合推理
    print("\n⏳ 正在分析知识图谱并生成答案...\n")
    result = await rag.aquery(query, param=QueryParam(mode="global"))
    
    print("\n📜 GraphRAG 回答：")
    print("-" * 70)
    print(result)
    print("-" * 70)
    
    # 额外的查询示例
    print("\n\n" + "=" * 70)
    print("📚 更多查询示例")
    print("=" * 70)
    
    queries = [
        "李白在哪些地方遇到过哪些诗人？",
        "杜甫担任过什么官职？",
        "贺知章对李白有什么评价？"
    ]
    
    for q in queries:
        print(f"\n🔍 查询：{q}")
        print("-" * 70)
        result = await rag.aquery(q)
        print(result[:500] if len(result) > 500 else result)
        print("-" * 70)

if __name__ == "__main__":
    asyncio.run(main())
