#!/usr/bin/env python3
"""测试自定义实体提取函数"""

import ollama
import nano_graphrag.prompt as prompts

def custom_entity_extract(text: str, entity_types: list, **kwargs):
    """自定义实体提取函数"""
    
    # 构建提示词
    prompt_template = prompts.PROMPTS['entity_extraction']
    filled_prompt = prompt_template.format(
        entity_types=entity_types,
        input_text=text,
        tuple_delimiter=prompts.PROMPTS['DEFAULT_TUPLE_DELIMITER'],
        record_delimiter=prompts.PROMPTS['DEFAULT_RECORD_DELIMITER'],
        completion_delimiter=prompts.PROMPTS['DEFAULT_COMPLETION_DELIMITER']
    )
    
    # 调用LLM
    response = ollama.generate(
        model="phi4-mini:3.8b",
        prompt=filled_prompt,
        options={'temperature': 0.1, 'num_predict': 1500}
    )
    
    result = response['response']
    
    # 确保格式正确
    if '<|COMPLETE|>' not in result:
        result = result.strip() + '\n<|COMPLETE|>'
    
    print(f"实体提取结果: {repr(result)}")
    
    return result

# 测试
test_text = "段誉是一位年轻的公子，他来到无量山剑湖宫观看比武。"
result = custom_entity_extract(test_text, ["person", "location", "event", "organization", "concept"])
print("\n最终结果:")
print(result)
