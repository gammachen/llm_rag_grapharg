#!/usr/bin/env python3
"""测试nano-graphrag默认配置"""

import asyncio
from nano_graphrag import GraphRAG

async def test_default_graphrag():
    """使用默认配置测试"""
    
    print("测试默认GraphRAG配置...")
    
    # 不传入任何自定义函数，使用默认配置
    graphrag = GraphRAG(
        working_dir="./test_default",
        enable_local=True,
    )
    
    # 测试文本
    test_text = "Duan Yu is a young nobleman who came to Wuliang Mountain Sword Lake Palace to watch the martial arts competition."
    
    # 插入文档
    await graphrag.ainsert(test_text)
    
    # 查询
    result = await graphrag.aquery("Who is Duan Yu?")
    print(f"查询结果: {result}")

if __name__ == "__main__":
    asyncio.run(test_default_graphrag())
