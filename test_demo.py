#!/usr/bin/env python3
"""
自动测试脚本 - 运行GraphRAG小说分析演示
"""

import asyncio
import os
import sys
from pathlib import Path

from novel_graphrag import NovelGraphRAG

async def test_graphrag():
    """测试GraphRAG功能"""
    
    print("=" * 60)
    print("🎯 GraphRAG小说分析系统 - 自动测试")
    print("=" * 60)
    print()
    
    novel_file = "./data/天龙八部-sample.txt"
    working_dir = "./novel_analysis_tianlong"
    
    if not os.path.exists(novel_file):
        print(f"❌ 测试文件不存在: {novel_file}")
        return
    
    print(f"📖 测试文件: {novel_file}")
    print(f"💾 工作目录: {working_dir}")
    print()
    
    novel_rag = NovelGraphRAG(
        working_dir=working_dir,
        llm_model="qwen3.5:9b",
        use_ollama_embedding=True,
        ollama_embedding_model="nomic-embed-text"
    )
    
    print("🔧 配置信息:")
    print(f"  - LLM模型: qwen3.5:9b")
    print(f"  - 嵌入模型: Ollama nomic-embed-text")
    print()
    
    print("📝 步骤1: 处理小说文本，构建知识图谱")
    print("-" * 60)
    graph_stats = await novel_rag.process_novel(novel_file)
    
    if graph_stats:
        print("\n📊 知识图谱统计:")
        for key, value in graph_stats.items():
            print(f"  • {key}: {value}")
    else:
        print("⚠️ 未能获取图谱统计信息")
    print()
    
    print("📝 步骤2: 执行查询测试")
    print("-" * 60)
    
    test_queries = [
        ("global", "这本小说的主要人物有哪些？他们的关系如何？"),
        ("local", "故事发生在什么地方？有什么重要的情节？"),
        ("naive", "简单描述一下故事的主要情节"),
    ]
    
    for mode, query in test_queries:
        print(f"\n🔍 查询 (模式: {mode}): {query}")
        print("-" * 60)
        try:
            result = await novel_rag.query_novel(query, mode=mode)
            print(result[:800] if len(result) > 800 else result)
        except Exception as e:
            print(f"⚠️ 查询出错: {e}")
        print()
    
    print("📝 步骤3: 导出数据")
    print("-" * 60)
    export_file = novel_rag.export_graph_data(
        os.path.join(working_dir, "test_graph_data.json")
    )
    
    viz_file = novel_rag.visualize_graph(
        os.path.join(working_dir, "test_knowledge_graph.png")
    )
    
    print()
    print("=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    
    if export_file:
        print(f"📄 图数据已导出: {export_file}")
    if viz_file:
        print(f"📊 可视化图谱: {viz_file}")

if __name__ == "__main__":
    asyncio.run(test_graphrag())
