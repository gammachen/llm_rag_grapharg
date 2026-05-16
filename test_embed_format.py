#!/usr/bin/env python3
"""测试嵌入函数返回格式"""

import numpy as np
import asyncio
import ollama

async def test_embedding():
    # 模拟nano-graphrag的批量调用
    texts = ["李白", "杜甫", "高适"]
    
    embeddings = []
    for text in texts:
        response = ollama.embeddings(
            model="nomic-embed-text:latest",
            prompt=text
        )
        embedding = np.array(response.embedding, dtype=np.float32)
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        embeddings.append(embedding)
    
    result = np.array(embeddings, dtype=np.float32)
    
    print(f"输入: {len(texts)} 个文本")
    print(f"返回形状: {result.shape}")
    print(f"返回维度: {result.ndim}")
    
    # 模拟np.concatenate
    try:
        concatenated = np.concatenate([result])
        print(f"concatenate形状: {concatenated.shape}")
        print("✅ concatenate成功")
    except Exception as e:
        print(f"❌ concatenate失败: {e}")
    
    # 测试单个文本
    single_result = result[0]
    print(f"\n单个向量形状: {single_result.shape}")
    
    try:
        single_concat = np.concatenate([single_result.reshape(1, -1)])
        print(f"单个concatenate形状: {single_concat.shape}")
        print("✅ 单个concatenate成功")
    except Exception as e:
        print(f"❌ 单个concatenate失败: {e}")

asyncio.run(test_embedding())
