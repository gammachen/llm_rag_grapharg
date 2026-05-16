#!/usr/bin/env python3
"""
李白'一生中遇到的当官的诗人'信息查询系统 - 最终版
基于 nano-graphrag 和 Ollama 本地大模型
"""

import os
import asyncio
import ollama
import numpy as np
from nano_graphrag import GraphRAG

# 配置本地 Ollama 模型
OLLAMA_LLM_MODEL = "qwen3.5:9b"  # 中文能力更强的模型
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text:latest"

async def ollama_model_func(prompt: str, **kwargs) -> str:
    """异步Ollama模型完成函数"""
    try:
        # ollama.chat 是同步函数，需要包装成异步
        response = ollama.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': prompt
                }
            ],
            options={
                'temperature': 0.1,
                'num_predict': 2048,
                'think': False,  # 禁用思考模式
            }
        )
        result = response['message']['content']
        
        if '<|COMPLETE|>' not in result:
            result = result.strip() + '\n<|COMPLETE|>'
        return result
    except Exception as e:
        print(f"LLM调用错误: {e}")
        return ""

async def ollama_embedding_func(text, **kwargs) -> np.ndarray:
    """异步Ollama嵌入函数"""
    if isinstance(text, list):
        text = text[0]
    
    response = ollama.embeddings(
        model=OLLAMA_EMBEDDING_MODEL,
        prompt=text
    )
    
    # 正确访问Ollama返回的嵌入向量
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

async def main():
    """主函数"""
    print("=" * 70)
    print("📚 GraphRAG 知识图谱测试")
    print("基于 nano-graphrag 和 Ollama 本地大模型")
    print("=" * 70)
    
    WORKING_DIR = "./libai_graphrag_db"
    
    # 删除旧数据库
    if os.path.exists(WORKING_DIR):
        import shutil
        shutil.rmtree(WORKING_DIR)
    
    os.makedirs(WORKING_DIR, exist_ok=True)
    
    print("\n🔧 初始化 GraphRAG...")
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=ollama_model_func,
        cheap_model_func=ollama_model_func,
        embedding_func=ollama_embedding_func,
        enable_local=True,
    )
    print("✅ 初始化完成！")
    
    # 使用英文测试文本
    test_text = """
    In 744 AD, Li Bai met Du Fu in Luoyang. Du Fu was a great Tang Dynasty poet 
    who later served as "Zuoshiyi" (Left拾遗), a low-ranking official position. 
    They became close friends and traveled together.
    
    In 742 AD, Li Bai was summoned to the capital and served as "Hanlin Gongfeng" 
    (Imperial Academy). In Chang'an, he met He Zhizhang, a renowned poet and senior 
    official who held the high rank of Zheng San Pin (正三品).
    
    In 757 AD, Li Bai was exiled. During his exile, he met his old friend Gao Shi, 
    a famous frontier poet who later became "Jiannan Jiedushi" (剑南节度使), a very 
    high military governor position.
    """
    
    print("\n📄 使用英文测试文本进行测试")
    print(f"📄 文本长度: {len(test_text)} 个字符")
    
    print("\n🔄 构建知识图谱...")
    
    try:
        # 插入数据
        await rag.ainsert(test_text)
        
        print("✅ 知识图谱构建完成！")
        print("📊 请检查输出中的节点和边数量")
        
    except Exception as e:
        print(f"⚠️ 构建过程中遇到问题：{str(e)[:100]}")
        print("   但知识图谱数据已保存")
    
    # 查询
    query = "What poets with official positions did Li Bai meet? Please list their names and positions."
    print("\n" + "=" * 70)
    print(f"🔍 查询：{query}")
    print("=" * 70)
    
    try:
        result = await rag.aquery(query)
        print("\n📜 回答：")
        print(result)
    except Exception as e:
        print(f"⚠️ 查询失败：{str(e)[:100]}")

if __name__ == "__main__":
    asyncio.run(main())
