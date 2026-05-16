#!/usr/bin/env python3
"""
小说GraphRAG测试应用
使用本地Ollama模型进行小说分析和知识图谱构建
"""

import asyncio
import os
import sys
from pathlib import Path

from novel_graphrag import NovelGraphRAG, AdvancedNovelAnalyzer


def find_test_data(data_dir: str = "./data") -> list:
    """查找data目录下的测试数据文件"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"⚠️ 数据目录不存在: {data_dir}")
        return []
    
    text_files = []
    for ext in ['*.txt', '*.md']:
        text_files.extend(data_path.glob(ext))
    
    return sorted(text_files)


def select_novel_file(data_dir: str = "./data") -> str:
    """让用户选择要处理的小说文件"""
    files = find_test_data(data_dir)
    
    if not files:
        print("❌ 未找到测试数据文件")
        return None
    
    print("\n📚 可用的测试数据文件:")
    print("-" * 50)
    for i, file_path in enumerate(files, 1):
        size = file_path.stat().st_size / 1024
        print(f"{i}. {file_path.name} ({size:.1f} KB)")
    print("-" * 50)
    
    while True:
        try:
            choice = input("\n请选择要处理的文件编号 (输入数字，或直接回车选择第一个): ").strip()
            
            if not choice:
                return str(files[0])
            
            idx = int(choice) - 1
            if 0 <= idx < len(files):
                return str(files[idx])
            else:
                print(f"⚠️ 无效选择，请输入 1-{len(files)} 之间的数字")
        except ValueError:
            print("⚠️ 请输入有效的数字")


async def process_and_analyze(novel_file: str, working_dir: str = "./novel_analysis"):
    """处理并分析小说"""
    
    print("=" * 60)
    print("🎯 小说GraphRAG分析系统")
    print("=" * 60)
    print(f"\n📖 处理文件: {novel_file}")
    print(f"💾 工作目录: {working_dir}")
    print()
    
    novel_rag = NovelGraphRAG(
        working_dir=working_dir,
        llm_model="qwen3:1.7b",
        embedding_model="sentence-transformers/all-MiniLM-L12-v2"
    )
    
    print("🔧 配置信息:")
    print(f"  - LLM模型: qwen3:1.7b")
    print(f"  - 嵌入模型: sentence-transformers/all-MiniLM-L12-v2")
    print(f"  - 工作目录: {working_dir}")
    print()
    
    graph_stats = await novel_rag.process_novel(novel_file)
    
    if graph_stats:
        print("\n📊 知识图谱统计:")
        for key, value in graph_stats.items():
            print(f"  • {key}: {value}")
    
    print("\n" + "=" * 60)
    print("💬 开始查询分析...")
    print("=" * 60)
    
    queries = [
        ("global", "这本小说的主要人物有哪些？他们的关系如何？"),
        ("local", "故事发生在什么地方？有什么重要的情节？"),
        ("naive", "简单描述一下故事的主要情节"),
    ]
    
    for mode, query in queries:
        print(f"\n🔍 查询 (模式: {mode}): {query}")
        print("-" * 60)
        try:
            result = await novel_rag.query_novel(query, mode=mode)
            print(result[:500] if len(result) > 500 else result)
        except Exception as e:
            print(f"⚠️ 查询出错: {e}")
        print()
    
    export_file = novel_rag.export_graph_data(
        os.path.join(working_dir, "novel_graph_data.json")
    )
    
    viz_file = novel_rag.visualize_graph(
        os.path.join(working_dir, "novel_knowledge_graph.png")
    )
    
    print("\n" + "=" * 60)
    print("✅ 分析完成！")
    print("=" * 60)
    
    return novel_rag


async def interactive_mode():
    """交互模式"""
    print("\n🎮 进入交互模式")
    print("输入问题进行查询，输入 'quit' 或 'exit' 退出")
    print("-" * 60)
    
    novel_file = select_novel_file()
    
    if not novel_file:
        return
    
    novel_name = Path(novel_file).stem
    working_dir = f"./novel_analysis_{novel_name}"
    
    novel_rag = await process_and_analyze(novel_file, working_dir)
    
    if not novel_rag.graphrag:
        print("\n⚠️ GraphRAG未初始化，无法进行查询")
        return
    
    print("\n🎮 交互式查询")
    print("-" * 60)
    
    while True:
        try:
            query = input("\n❓ 请输入您的问题 (输入 'quit' 退出): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 再见！")
                break
            
            if not query:
                continue
            
            mode = input("选择查询模式 [global/local/naive] (直接回车使用 global): ").strip() or "global"
            
            if mode not in ['global', 'local', 'naive']:
                print("⚠️ 无效模式，使用默认的 global")
                mode = "global"
            
            print(f"\n🔍 查询: {query}")
            print("-" * 60)
            
            result = await novel_rag.query_novel(query, mode=mode)
            print(result)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n⚠️ 查询出错: {e}")


async def demo_mode():
    """演示模式 - 使用默认数据"""
    
    print("\n🎬 进入演示模式")
    print("-" * 60)
    
    data_dir = "./data"
    files = find_test_data(data_dir)
    
    if not files:
        print("❌ 未找到测试数据")
        return
    
    novel_file = str(files[0])
    print(f"📖 使用测试数据: {novel_file}")
    
    novel_name = Path(novel_file).stem
    working_dir = f"./novel_analysis_{novel_name}"
    
    await process_and_analyze(novel_file, working_dir)


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("📚 小说GraphRAG分析系统")
    print("基于nano-graphrag和本地Ollama模型")
    print("=" * 60)
    
    print("\n请选择运行模式:")
    print("1. 演示模式 (使用默认测试数据)")
    print("2. 交互模式 (选择数据文件)")
    print("3. 快速测试 (仅处理数据，不进行查询)")
    
    while True:
        try:
            choice = input("\n请选择 (1/2/3): ").strip()
            
            if choice == '1':
                await demo_mode()
                break
            elif choice == '2':
                await interactive_mode()
                break
            elif choice == '3':
                novel_file = select_novel_file()
                if novel_file:
                    novel_name = Path(novel_file).stem
                    working_dir = f"./novel_analysis_{novel_name}"
                    
                    print("\n🔄 正在处理小说...")
                    novel_rag = NovelGraphRAG(
                        working_dir=working_dir,
                        llm_model="qwen3.5:9b"
                    )
                    
                    await novel_rag.process_novel(novel_file)
                    print(f"\n✅ 处理完成！数据保存在: {working_dir}")
                break
            else:
                print("⚠️ 无效选择，请输入 1、2 或 3")
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n⚠️ 出错: {e}")


if __name__ == "__main__":
    print("🚀 启动应用...")
    
    try:
        import ollama
        print("✅ Ollama 已安装")
        
        try:
            models = ollama.list()
            print(f"📦 可用的Ollama模型: {len(models.get('models', []))} 个")
        except:
            print("⚠️ 无法连接到Ollama服务，请确保Ollama正在运行")
    
    except ImportError:
        print("⚠️ Ollama Python库未安装，将使用模拟功能")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers 已安装")
    except ImportError:
        print("⚠️ sentence-transformers 未安装，将使用模拟嵌入")
    
    try:
        from nano_graphrag import GraphRAG
        print("✅ nano-graphrag 已安装")
    except ImportError:
        print("⚠️ nano-graphrag 未安装，将使用简化模式")
    
    print("\n" + "-" * 60)
    
    asyncio.run(main())
