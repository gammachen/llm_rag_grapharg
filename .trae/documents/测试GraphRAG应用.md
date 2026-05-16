# GraphRAG小说分析应用测试计划

## 任务目标
在现有conda环境中安装依赖并运行测试脚本，验证GraphRAG小说分析应用的可行性。

## 实施步骤

### 1. 环境准备
- [ ] 检查现有conda环境配置
- [ ] 确认Python版本

### 2. 依赖安装
- [ ] 安装requirements.txt中的核心依赖：
  - nano-graphrag
  - faiss-cpu
  - sentence-transformers
  - networkx
  - matplotlib
  - numpy
  - python-dotenv
  - ollama (Python SDK)

### 3. Ollama服务检查
- [ ] 检查Ollama服务是否运行
- [ ] 验证qwen3.5:9b模型是否可用
- [ ] 验证嵌入模型配置

### 4. 运行测试
- [ ] 执行python app.py脚本
- [ ] 选择测试模式（演示模式/交互模式）
- [ ] 处理测试数据（天龙八部-sample.txt）
- [ ] 验证知识图谱构建
- [ ] 执行查询测试

### 5. 结果记录
- [ ] 记录成功/失败情况
- [ ] 记录遇到的问题和解决方案
- [ ] 给出性能评估

## 预期输出
- 成功构建知识图谱
- 能够进行三种模式的查询（global/local/naive）
- 生成可视化图谱文件
- 导出JSON格式的图数据

## 风险评估
- Ollama服务未运行 → 需要先启动服务
- 模型未下载 → 需要先下载qwen3.5:9b
- 依赖冲突 → 可能需要创建新环境
- 内存不足 → 可能需要使用更小的模型

## 测试数据
- 使用 `/Users/shhaofu/Code/cursor-projects/llm_rag_grapharg/data/天龙八部-sample.txt`
- 约10KB，包含约50行文本
