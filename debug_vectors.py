#!/usr/bin/env python3
"""详细调试nano-graphrag的向量处理"""

import asyncio
import ollama
import numpy as np

OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

async def ollama_embedding_func(text, **kwargs) -> np.ndarray:
    """异步Ollama嵌入函数"""
    if isinstance(text, list):
        text = text[0]
    
    print(f"\n=== 嵌入函数被调用 ===")
    print(f"输入类型: {type(text)}")
    print(f"输入内容前100字符: {str(text)[:100]}")
    
    response = ollama.embeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        prompt=text
    )
    
    print(f"\n响应类型: {type(response)}")
    print(f"响应内容类型: {type(response.embedding)}")
    print(f"响应内容长度: {len(response.embedding) if hasattr(response.embedding, '__len__') else 'N/A'}")
    
    embedding = np.array(response.embedding, dtype=np.float32)
    
    print(f"numpy数组类型: {type(embedding)}")
    print(f"numpy数组形状: {embedding.shape}")
    print(f"numpy数组维度: {embedding.ndim}")
    print(f"前5个元素: {embedding[:5]}")
    
    # 确保返回正确的形状
    if embedding.ndim == 0:
        print("警告：数组是标量，创建零向量")
        embedding = np.zeros(768, dtype=np.float32)
    elif embedding.ndim > 1:
        print(f"警告：数组维度大于1，flatten")
        embedding = embedding.flatten()
    
    print(f"最终返回形状: {embedding.shape}")
    print("=== 嵌入函数结束 ===\n")
    
    return embedding

async def test_nano_graphrag():
    """测试nano-graphrag的向量处理"""
    from nano_graphrag import GraphRAG
    
    async def ollama_model_func(prompt: str, **kwargs) -> str:
        response = ollama.generate(
            model="phi4-mini:3.8b",
            prompt=prompt,
            options={'temperature': 0.1, 'num_predict': 2048}
        )
        result = response['response']
        if '<|COMPLETE|>' not in result:
            result = result.strip() + '\n<|COMPLETE|>'
        return result
    
    ollama_embedding_func.embedding_dim = 768
    
    print("初始化GraphRAG...")
    rag = GraphRAG(
        working_dir="./libai_graphrag_debug",
        best_model_func=ollama_model_func,
        cheap_model_func=ollama_model_func,
        embedding_func=ollama_embedding_func,
        enable_local=True,
    )
    print("初始化完成\n")
    
    # 测试文本
    test_text = "李白在洛阳遇到了杜甫。"
    
    print("插入测试文本...")
    try:
        await rag.ainsert(test_text)
        print("\n✅ 插入成功！")
    except Exception as e:
        print(f"\n❌ 插入失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_nano_graphrag())
