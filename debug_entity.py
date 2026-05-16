#!/usr/bin/env python3
"""调试nano-graphrag实体提取"""

import ollama
import nano_graphrag.prompt as prompts

# 查看默认的实体类型和分隔符
print("默认实体类型:", prompts.PROMPTS['DEFAULT_ENTITY_TYPES'])
print("默认元组分隔符:", repr(prompts.PROMPTS['DEFAULT_TUPLE_DELIMITER']))
print("默认记录分隔符:", repr(prompts.PROMPTS['DEFAULT_RECORD_DELIMITER']))
print("默认完成分隔符:", repr(prompts.PROMPTS['DEFAULT_COMPLETION_DELIMITER']))

# 查看实体提取提示词模板
prompt_template = prompts.PROMPTS['entity_extraction']
print("\n实体提取提示词模板 (前1000字符):")
print("-" * 60)
print(prompt_template[:1000])
print("-" * 60)

# 测试实际的提示词填充
test_text = "段誉是一位年轻的公子，他来到无量山剑湖宫观看比武。"
filled_prompt = prompt_template.format(
    entity_types=["person", "location", "event", "organization", "concept"],
    input_text=test_text,
    tuple_delimiter=prompts.PROMPTS['DEFAULT_TUPLE_DELIMITER'],
    record_delimiter=prompts.PROMPTS['DEFAULT_RECORD_DELIMITER'],
    completion_delimiter=prompts.PROMPTS['DEFAULT_COMPLETION_DELIMITER']
)

print("\n填充后的提示词:")
print("-" * 60)
print(filled_prompt)
print("-" * 60)

# 调用LLM
print("\n调用LLM...")
try:
    response = ollama.generate(
        model="phi4-mini:3.8b",
        prompt=filled_prompt,
        options={'temperature': 0.1, 'num_predict': 1500}
    )
    
    print("\nLLM返回结果:")
    print("-" * 60)
    print(repr(response['response']))
    print("-" * 60)
    
except Exception as e:
    print(f"Error: {e}")
