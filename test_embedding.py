#!/usr/bin/env python3
"""测试nano-graphrag的嵌入函数调用方式"""

import asyncio
import ollama
import numpy as np

# 测试嵌入函数
async def test_embedding():
    """测试Ollama嵌入"""
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt="测试文本"
    )
    
    print(f"响应类型: {type(response)}")
    print(f"响应键: {response.keys()}")
    print(f"embedding类型: {type(response['embedding'])}")
    print(f"embedding长度: {len(response['embedding'])}")
    
    embedding = np.array(response['embedding'], dtype=np.float32)
    print(f"numpy数组形状: {embedding.shape}")
    print(f"numpy数组维度: {embedding.ndim}")
    
    # 测试返回列表
    return [embedding.tolist()]

# 运行测试
result = asyncio.run(test_embedding())
print(f"\n返回结果: {type(result)}")
print(f"结果长度: {len(result)}")
print(f"第一个元素类型: {type(result[0])}")
