#!/usr/bin/env python3
"""完整的GraphRAG测试 - 修复向量问题"""

import asyncio
import ollama
import numpy as np
from nano_graphrag import GraphRAG

async def ollama_complete(prompt: str, **kwargs) -> str:
    """异步Ollama模型完成函数"""
    response = ollama.generate(
        model="phi4-mini:3.8b",
        prompt=prompt,
        options={'temperature': 0.1, 'num_predict': 2048}
    )
    result = response['response']
    if '<|COMPLETE|>' not in result:
        result = result.strip() + '\n<|COMPLETE|>'
    return result

async def ollama_embedding(text, **kwargs) -> np.ndarray:
    """异步Ollama嵌入函数"""
    if isinstance(text, list):
        text = text[0]
    
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    
    embedding = np.array(response['embedding'], dtype=np.float32)
    
    # 确保返回2D数组
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    return embedding

async def test_graphrag():
    """测试GraphRAG"""
    
    print("=" * 60)
    print("🎯 GraphRAG测试 - 使用Ollama模型")
    print("=" * 60)
    
    ollama_embedding.embedding_dim = 768
    
    graphrag = GraphRAG(
        working_dir="./novel_analysis_final",
        best_model_func=ollama_complete,
        cheap_model_func=ollama_complete,
        embedding_func=ollama_embedding,
        enable_local=True,
        embedding_batch_num=1,
    )
    
    test_text = """
Duan Yu is a young nobleman who came to Wuliang Mountain Sword Lake Palace 
to watch the martial arts competition. Zuo Zimu is the leader of the Eastern 
Sect of Wuliang Sword. Gong Guangjie is Zuo Zimu's disciple.
"""
    
    print("\n📝 测试文本:")
    print(test_text.strip())
    print("\n🔄 插入文档...")
    
    await graphrag.ainsert(test_text)
    
    print("\n✅ 文档插入完成!")
    print("\n🔍 执行查询...")
    
    result = await graphrag.aquery("Who is Duan Yu?")
    print(f"\n查询结果:\n{result}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_graphrag())
