#!/usr/bin/env python3
"""
李白生平信息查询系统 - 修复查询功能
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
    print("基于 nano-graphrag 和 Ollama 本地大模型")
    print("=" * 70)
    
    WORKING_DIR = "./libai_graphrag_db"
    
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
    
    # 直接检查图谱数据
    print("=" * 70)
    print("📊 知识图谱数据分析")
    print("=" * 70)
    
    graph_db = rag._chunk_entity_relation_graph
    
    # 获取所有节点
    all_nodes = await graph_db.get_all_nodes()
    print(f"\n📝 实体节点: {len(all_nodes)} 个")
    print("-" * 70)
    for node_name in sorted(all_nodes.keys())[:15]:
        node_data = await graph_db.get_node(node_name)
        if node_data:
            entity_type = node_data.get('entity_type', 'N/A')
            desc = node_data.get('description', 'N/A')
            print(f"  • {node_name}")
            print(f"    类型: {entity_type}")
            print(f"    描述: {desc[:80]}...")
            print()
    
    # 获取所有边
    all_edges = await graph_db.get_all_edges()
    print(f"\n🔗 关系边: {len(all_edges)} 条")
    print("-" * 70)
    for edge_name in sorted(all_edges.keys())[:10]:
        edge_data = await graph_db.get_edge(edge_name)
        if edge_data:
            desc = edge_data.get('description', 'N/A')
            weight = edge_data.get('weight', 'N/A')
            print(f"  • {edge_name[0]} → {edge_name[1]}")
            print(f"    描述: {desc[:80]}...")
            print(f"    权重: {weight}")
            print()
    
    # 尝试查询
    query = "Which poets who held official positions did Li Bai encounter? List their names and positions."
    print("\n" + "=" * 70)
    print(f"🔍 查询：{query}")
    print("=" * 70)
    
    try:
        # 使用 local 模式（不依赖社区报告）
        result = await rag.aquery(query, param=QueryParam(mode="local"))
        print("\n📜 回答：")
        print(result)
    except Exception as e:
        print(f"\n⚠️ 查询错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
