#!/usr/bin/env python3
"""调试nano-graphrag实体解析"""

import ollama
import nano_graphrag.prompt as prompts

# 测试数据
test_text = "段誉是一位年轻的公子，他来到无量山剑湖宫观看比武。"

# 构建完整提示词
prompt_template = prompts.PROMPTS['entity_extraction']
filled_prompt = prompt_template.format(
    entity_types=["person", "location", "event", "organization", "concept"],
    input_text=test_text,
    tuple_delimiter=prompts.PROMPTS['DEFAULT_TUPLE_DELIMITER'],
    record_delimiter=prompts.PROMPTS['DEFAULT_RECORD_DELIMITER'],
    completion_delimiter=prompts.PROMPTS['DEFAULT_COMPLETION_DELIMITER']
)

# 调用LLM
print("调用LLM...")
response = ollama.generate(
    model="phi4-mini:3.8b",
    prompt=filled_prompt,
    options={'temperature': 0.1, 'num_predict': 1500}
)

llm_output = response['response']
print(f"LLM原始输出: {repr(llm_output)}")

# 添加COMPLETE标记
if '<|COMPLETE|>' not in llm_output:
    llm_output = llm_output.strip() + '<|COMPLETE|>'
print(f"添加标记后: {repr(llm_output)}")

# 测试nano-graphrag的解析函数
from nano_graphrag.parser import parse_entities_and_relations

print("\n测试nano-graphrag解析...")
try:
    entities, relations = parse_entities_and_relations(
        llm_output,
        tuple_delimiter=prompts.PROMPTS['DEFAULT_TUPLE_DELIMITER'],
        record_delimiter=prompts.PROMPTS['DEFAULT_RECORD_DELIMITER'],
        completion_delimiter=prompts.PROMPTS['DEFAULT_COMPLETION_DELIMITER']
    )
    
    print(f"解析出的实体: {len(entities)}")
    for e in entities:
        print(f"  - {e}")
    
    print(f"解析出的关系: {len(relations)}")
    for r in relations:
        print(f"  - {r}")
        
except Exception as e:
    print(f"解析错误: {e}")
    import traceback
    traceback.print_exc()
