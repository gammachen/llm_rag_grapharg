#!/usr/bin/env python3
"""
李白生平信息查询系统 - 测试查询功能
基于 nano-graphrag 和 Ollama 本地大模型
"""

import os
import asyncio
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam

# 配置本地 Ollama 模型
OLLAMA_LLM_MODEL = "phi4-mini:3.8b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"

async def ollama_model_func(prompt: str, **kwargs) -> str:
    """异步Ollama模型完成函数"""
    try:
        messages = []
        history_messages = kwargs.get('history_messages', [])
        if history_messages:
            for msg in history_messages:
                messages.append(msg)
        
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        response = ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=messages,
            options={
                'temperature': 0.1,
                'num_predict': 3000,
            }
        )
        result = response['message']['content']
        
        if '<|COMPLETE|>' not in result:
            result = result.strip() + '\n<|COMPLETE|>'
        return result
    except Exception as e:
        print(f"LLM调用错误: {e}")
        return ""

async def ollama_embedding_func(texts, **kwargs) -> np.ndarray:
    """异步Ollama嵌入函数"""
    if isinstance(texts, str):
        text_list = [texts]
    else:
        text_list = list(texts)
    
    embeddings = []
    for text in text_list:
        response = ollama.embeddings(
            model=OLLAMA_EMBEDDING_MODEL,
            prompt=text
        )
        embedding = np.array(response.embedding, dtype=np.float32)
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        embeddings.append(embedding)
    
    result = np.array(embeddings, dtype=np.float32)
    
    if result.ndim == 1:
        result = result.reshape(1, -1)
    
    return result

ollama_embedding_func.embedding_dim = 768

async def main():
    """主函数"""
    print("=" * 70)
    print("📚 李白生平信息查询系统")
    print("=" * 70)
    
    WORKING_DIR = "./libai_graphrag_db"
    
    print("\n🔧 初始化 GraphRAG...")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_func,
        cheap_model_func=ollama_model_func,
        embedding_func=ollama_embedding_func,
    )
    print("✅ 初始化完成！\n")
    
    # 尝试查询
    query = "Which poets who held official positions did Li Bai encounter? List their names and positions."
    print("=" * 70)
    print(f"🔍 查询：{query}")
    print("=" * 70)
    
    try:
        result = await rag.aquery(query, param=QueryParam(mode="local"))
        print("\n📜 回答：")
        print(result)
    except Exception as e:
        print(f"\n⚠️ 查询错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
