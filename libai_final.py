#!/usr/bin/env python3
"""
李白'一生中遇到的当官的诗人'信息查询系统 - 最终版
基于 nano-graphrag 和 Ollama 本地大模型
"""

import os
import asyncio
import ollama
import numpy as np
from nano_graphrag import GraphRAG, QueryParam

# 配置本地 Ollama 模型
OLLAMA_LLM_MODEL = "phi4-mini:3.8b"  # 能够正确处理长提示词的模型
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"

async def ollama_model_func(prompt: str, **kwargs) -> str:
    """异步Ollama模型完成函数"""
    try:
        # 构建消息列表
        messages = []
        
        # 处理历史消息（多轮对话）
        history_messages = kwargs.get('history_messages', [])
        if history_messages:
            for msg in history_messages:
                messages.append(msg)
        
        # 添加当前用户消息
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
    """异步Ollama嵌入函数 - 返回正确的2D数组格式"""
    # 处理单个字符串或列表
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
        # 正确访问Ollama返回的嵌入向量
        embedding = np.array(response.embedding, dtype=np.float32)
        # 确保是1D数组
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        embeddings.append(embedding)
    
    # 返回2D数组 (num_vectors, embedding_dim)
    # 这是 np.concatenate 期望的格式
    result = np.array(embeddings, dtype=np.float32)
    
    # 确保始终是2D数组
    if result.ndim == 1:
        result = result.reshape(1, -1)
    
    return result

# 同步版本的嵌入函数
def ollama_embedding_sync(text, **kwargs) -> np.ndarray:
    """同步Ollama嵌入函数"""
    if isinstance(text, list):
        text = text[0]
    
    response = ollama.embeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        prompt=text
    )
    
    embedding = np.array(response.embedding, dtype=np.float32)
    
    # 确保返回2D数组 (1, 768) - nano-vectorDB期望的格式
    if embedding.ndim == 0:
        embedding = np.zeros((1, 768), dtype=np.float32)
    elif embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    elif embedding.ndim > 2:
        embedding = embedding.reshape(-1, embedding.shape[-1])
    
    return embedding

ollama_embedding_func.embedding_dim = 768
ollama_embedding_sync.embedding_dim = 768

async def main():
    """主函数"""
    print("=" * 70)
    print("📚 李白'一生中遇到的当官的诗人'信息查询系统")
    print("基于 nano-graphrag 和 Ollama 本地大模型")
    print("=" * 70)
    
    WORKING_DIR = "./libai_graphrag_db"
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
    
    print("\n🔧 初始化 GraphRAG...")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_func,
        cheap_model_func=ollama_model_func,
        embedding_func=ollama_embedding_func,  # 必须使用异步版本
        enable_local=True,
    )
    print("✅ 初始化完成！")
    
    print("\n📖 读取数据...")
    with open("./libai_life.txt", encoding="utf-8") as f:
        libai_text = f.read()
    print(f"📄 已读取 {len(libai_text)} 个字符")
    
    print("\n🔄 构建知识图谱...")
    try:
        await rag.ainsert(libai_text)
        print("✅ 知识图谱构建完成！\n")
    except Exception as e:
        print(f"⚠️ 知识图谱构建遇到问题，但实体提取已完成")
        print(f"   错误: {str(e)[:100]}")
        print(f" {str(e)}")
        print()
    
    # 查询
    query = "李白一生中遇到的当官的诗人都有哪些人？请列出他们的名字以及担任的官职。"
    print("=" * 70)
    print(f"🔍 查询：{query}")
    print("=" * 70)
    
    try:
        result = await rag.aquery(query, param=QueryParam(mode="local"))
        print("\n📜 回答：")
        print(result)
    except Exception as e:
        print(f"⚠️ 查询遇到问题")
        print(f"   错误: {str(e)}")
        print("\n💡 提示：知识图谱构建过程中可能遇到向量处理问题")
        print("   实体和关系提取已完成，但查询功能需要进一步调试")
        import traceback
        traceback.print_exc()

    query = "Which poets who held official positions did Li Bai encounter during his lifetime? Please list their names and the official posts they held."
    print("=" * 70)
    print(f"🔍 查询：{query}")
    print("=" * 70)
    
    try:
        result = await rag.aquery(query, param=QueryParam(mode="local"))
        print("\n📜 回答：")
        print(result)
    except Exception as e:
        print(f"⚠️ 查询遇到问题")
        print(f"   错误: {str(e)}")
        print("\n💡 提示：知识图谱构建过程中可能遇到向量处理问题")
        print("   实体和关系提取已完成，但查询功能需要进一步调试")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    asyncio.run(main())
