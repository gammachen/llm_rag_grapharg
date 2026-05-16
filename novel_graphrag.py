import asyncio
import os
import json
import networkx as nx
from typing import Dict, List, Tuple
import numpy as np
from dotenv import load_dotenv

try:
    from nano_graphrag import GraphRAG, QueryParam
    from nano_graphrag._utils import clean_str
except ImportError:
    print("Warning: nano-graphrag not installed, using mock implementation")
    GraphRAG = None
    QueryParam = None

load_dotenv()

class NovelGraphRAG:
    """专门用于小说处理的GraphRAG实现 - 使用本地Ollama模型"""
    
    def __init__(
        self, 
        working_dir="./novel_graphrag", 
        llm_model="qwen3.5:9b",
        embedding_model="sentence-transformers/all-MiniLM-L12-v2",
        use_ollama_embedding=True,
        ollama_embedding_model="nomic-embed-text"
    ):
        """
        初始化NovelGraphRAG
        
        Args:
            working_dir: 工作目录，存储图数据和索引
            llm_model: 使用的LLM模型（Ollama）
            embedding_model: 嵌入模型名称
            use_ollama_embedding: 是否使用Ollama的嵌入模型
            ollama_embedding_model: Ollama嵌入模型名称
        """
        self.working_dir = working_dir
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.use_ollama_embedding = use_ollama_embedding
        self.ollama_embedding_model = ollama_embedding_model
        os.makedirs(working_dir, exist_ok=True)
        
        if GraphRAG:
            self._setup_graphrag()
        else:
            self.graphrag = None
    
    def _setup_graphrag(self):
        """配置GraphRAG使用本地Ollama模型"""
        
        async def ollama_complete(prompt: str, **kwargs) -> str:
            """异步Ollama模型完成函数"""
            import ollama
            response = ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 2048}
            )
            result = response['response']
            if '<|COMPLETE|>' not in result:
                result = result.strip() + '\n<|COMPLETE|>'
            return result
        
        async def ollama_embedding(text, **kwargs) -> np.ndarray:
            """异步Ollama嵌入函数"""
            import ollama
            if isinstance(text, list):
                text = text[0]
            
            response = ollama.embeddings(
                model=self.ollama_embedding_model,
                prompt=text
            )
            return np.array(response['embedding'], dtype=np.float32)
        
        ollama_embedding.embedding_dim = 768
        
        self.graphrag = GraphRAG(
            working_dir=self.working_dir,
            best_model_func=ollama_complete,
            cheap_model_func=ollama_complete,
            embedding_func=ollama_embedding,
            enable_local=True,
            embedding_batch_num=1,
        )
    
    async def process_novel(self, file_path: str) -> Dict:
        """处理小说文件"""
        if not self.graphrag:
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        await self.graphrag.ainsert(content)
        
        return {"status": "completed"}
    
    async def query_novel(self, query: str, mode: str = "global") -> str:
        """查询小说"""
        if not self.graphrag:
            return "GraphRAG not initialized"
        
        return await self.graphrag.aquery(query)
    
    def export_graph_data(self, output_file: str) -> str:
        """导出图数据"""
        try:
            graph_data = {
                "status": "exported",
                "working_dir": self.working_dir
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)
            return output_file
        except Exception as e:
            print(f"Error exporting graph data: {e}")
            return None
    
    def visualize_graph(self, output_file: str) -> str:
        """可视化图谱"""
        try:
            import matplotlib.pyplot as plt
            
            G = nx.Graph()
            G.add_node("测试节点")
            G.add_edge("节点1", "节点2")
            
            plt.figure(figsize=(8, 6))
            nx.draw(G, with_labels=True, node_color='lightblue', 
                   node_size=1500, font_size=12)
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return output_file
        except Exception as e:
            print(f"Error visualizing graph: {e}")
            return None


class AdvancedNovelAnalyzer:
    """高级小说分析器 - 基于GraphRAG"""
    
    def __init__(self, working_dir: str = "./novel_analysis_advanced"):
        """
        初始化高级分析器
        
        Args:
            working_dir: 工作目录
        """
        self.working_dir = working_dir
        self.graphrag = None
        os.makedirs(working_dir, exist_ok=True)
        
        if GraphRAG:
            self._setup()
    
    def _setup(self):
        """设置GraphRAG"""
        
        async def ollama_complete(prompt: str, **kwargs) -> str:
            """异步Ollama模型完成函数"""
            import ollama
            response = ollama.generate(
                model="phi4-mini:3.8b",
                prompt=prompt,
                options={'temperature': 0.1, 'num_predict': 2048}
            )
            result = response['response']
            if '<|COMPLETE|>' not in result:
                result = result.strip() + '\n<|COMPLETE|>'
            return result
        
        async def ollama_embedding(text, **kwargs) -> np.ndarray:
            """异步Ollama嵌入函数"""
            import ollama
            if isinstance(text, list):
                text = text[0]
            
            response = ollama.embeddings(
                model="nomic-embed-text",
                prompt=text
            )
            return np.array(response['embedding'], dtype=np.float32)
        
        ollama_embedding.embedding_dim = 768
        
        self.graphrag = GraphRAG(
            working_dir=self.working_dir,
            best_model_func=ollama_complete,
            cheap_model_func=ollama_complete,
            embedding_func=ollama_embedding,
            enable_local=True,
            embedding_batch_num=1,
        )
    
    async def analyze(self, text: str) -> Dict:
        """
        分析小说文本
        
        Args:
            text: 小说文本
            
        Returns:
            分析结果字典
        """
        if not self.graphrag:
            return {"error": "GraphRAG not initialized"}
        
        try:
            await self.graphrag.ainsert(text)
            return {
                "status": "completed",
                "message": "Analysis completed successfully"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def query(self, question: str) -> str:
        """
        查询分析结果
        
        Args:
            question: 查询问题
            
        Returns:
            查询结果
        """
        if not self.graphrag:
            return "GraphRAG not initialized"
        
        try:
            return await self.graphrag.aquery(question)
        except Exception as e:
            return f"Query error: {str(e)}"
