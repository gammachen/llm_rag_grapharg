#!/usr/bin/env python3
"""最终测试脚本 - 使用phi4-mini模型"""

import asyncio
from novel_graphrag import NovelGraphRAG

async def test_final():
    """最终测试"""
    
    print("=" * 60)
    print("🎯 GraphRAG小说分析系统 - 最终测试")
    print("=" * 60)
    
    novel_file = "./data/天龙八部-sample.txt"
    working_dir = "./novel_analysis_test_final"
    
    novel_rag = NovelGraphRAG(
        working_dir=working_dir,
        llm_model="phi4-mini:3.8b",
        use_ollama_embedding=True,
        ollama_embedding_model="nomic-embed-text"
    )
    
    print(f"\n📖 测试文件: {novel_file}")
    print(f"💾 工作目录: {working_dir}")
    print(f"🔧 LLM模型: phi4-mini:3.8b")
    print(f"🔧 嵌入模型: nomic-embed-text")
    
    print("\n📝 开始处理小说...")
    
    try:
        result = await novel_rag.process_novel(novel_file)
        print("\n✅ 小说处理完成!")
        
        print("\n🔍 执行查询...")
        query_result = await novel_rag.query_novel("故事的主要人物有哪些?")
        print(f"\n查询结果:\n{query_result}")
        
    except Exception as e:
        print(f"\n⚠️ 测试出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试结束")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_final())
