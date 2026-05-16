#!/usr/bin/env python3
"""
李白'一生中遇到的当官的诗人'信息查询系统 - 简化版
基于 nano-graphrag 和 Ollama 本地大模型
"""

import os
import asyncio
import ollama
import numpy as np
from nano_graphrag import GraphRAG

# 配置本地 Ollama 模型
OLLAMA_LLM_MODEL = "phi4-mini:3.8b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

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

async def ollama_embedding_func(text, **kwargs) -> np.ndarray:
    """异步Ollama嵌入函数"""
    if isinstance(text, list):
        text = text[0]
    
    response = ollama.embeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        prompt=text
    )
    return np.array(response['embedding'], dtype=np.float32)

ollama_embedding_func.embedding_dim = 768

async def main():
    """主函数"""
    print("=" * 70)
    print("📚 李白'一生中遇到的当官的诗人'信息查询系统")
    print("=" * 70)
    
    WORKING_DIR = "./libai_graphrag_db"
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    
    print("\n🔧 初始化 GraphRAG...")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_func,
        cheap_model_func=ollama_model_func,
        embedding_func=ollama_embedding_func,
        enable_local=True,
    )
    print("✅ 初始化完成！")
    
    print("\n📖 读取数据...")
    with open("./libai_life.txt", encoding="utf-8") as f:
        libai_text = f.read()
    print(f"📄 已读取 {len(libai_text)} 个字符")
    
    print("\n🔄 构建知识图谱...")
    await rag.ainsert(libai_text)
    print("✅ 知识图谱构建完成！\n")
    
    # 查询
    query = "李白一生中遇到的当官的诗人都有哪些人？请列出他们的名字以及担任的官职。"
    print("=" * 70)
    print(f"🔍 查询：{query}")
    print("=" * 70)
    
    result = await rag.aquery(query)
    print("\n📜 回答：")
    print(result)
    
    print("\n\n📚 更多查询：")
    print("-" * 70)
    
    queries = [
        "李白在哪些地方遇到过哪些诗人？",
        "杜甫担任过什么官职？"
    ]
    
    for q in queries:
        print(f"\n🔍 {q}")
        print(await rag.aquery(q)[:300])
        print("-" * 70)

if __name__ == "__main__":
    asyncio.run(main())
