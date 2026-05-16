#!/usr/bin/env python3
"""测试Ollama嵌入的正确访问方式"""

import asyncio
import ollama
import numpy as np

async def test_embedding():
    """测试Ollama嵌入"""
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt="测试文本"
    )
    
    print(f"响应类型: {type(response)}")
    print(f"响应: {response}")
    
    # 正确的访问方式
    embedding = response.embedding
    print(f"\nembedding类型: {type(embedding)}")
    print(f"embedding长度: {len(embedding)}")
    
    np_embedding = np.array(embedding, dtype=np.float32)
    print(f"numpy数组形状: {np_embedding.shape}")
    
    return np_embedding

# 运行测试
result = asyncio.run(test_embedding())
print(f"\n返回结果形状: {result.shape}")
