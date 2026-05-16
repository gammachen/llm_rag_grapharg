#!/usr/bin/env python3
"""测试使用同步嵌入函数的GraphRAG"""

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

def ollama_embedding_sync(text, **kwargs) -> np.ndarray:
    """同步Ollama嵌入函数"""
    if isinstance(text, list):
        text = text[0]
    
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    
    return np.array(response['embedding'], dtype=np.float32)

async def test_ollama_graphrag():
    """测试使用Ollama模型"""
    
    print("测试Ollama模型的GraphRAG...")
    
    ollama_embedding_sync.embedding_dim = 768
    
    graphrag = GraphRAG(
        working_dir="./test_ollama_sync",
        best_model_func=ollama_complete,
        cheap_model_func=ollama_complete,
        embedding_func=ollama_embedding_sync,
        enable_local=True,
    )
    
    test_text = "Duan Yu is a young nobleman."
    
    print("插入文档...")
    await graphrag.ainsert(test_text)
    
    print("查询...")
    result = await graphrag.aquery("Who is Duan Yu?", mode="local")
    print(f"查询结果: {result}")

if __name__ == "__main__":
    asyncio.run(test_ollama_graphrag())
