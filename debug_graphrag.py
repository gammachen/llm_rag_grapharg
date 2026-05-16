#!/usr/bin/env python3
"""查看nano-graphrag的模型调用方式"""

from nano_graphrag import GraphRAG
import inspect

# 查看GraphRAG的方法
print("GraphRAG方法列表:")
print([m for m in dir(GraphRAG) if not m.startswith('_')])

# 查看process方法
print("\nprocess方法签名:")
print(inspect.signature(GraphRAG.process))

# 查看默认模型函数
print("\n查看默认模型函数...")
try:
    import nano_graphrag.llm
    print("nano_graphrag.llm模块内容:", dir(nano_graphrag.llm))
except:
    pass

# 测试直接调用
print("\n创建GraphRAG实例并测试...")
g = GraphRAG(working_dir='/tmp/test_graph')
print("GraphRAG属性:", [attr for attr in dir(g) if not attr.startswith('_')])
