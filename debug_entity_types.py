#!/usr/bin/env python3
"""检查实体类型格式"""

import nano_graphrag.prompt as prompts

# 查看提示词模板中的实体类型格式
prompt_template = prompts.PROMPTS['entity_extraction']

# 找到包含entity_type的部分
lines = prompt_template.split('\n')
for i, line in enumerate(lines):
    if 'entity_type' in line.lower():
        print(f"行 {i}: {line}")

# 测试不同格式的实体类型
entity_types1 = ["person", "location", "event", "organization", "concept"]
entity_types2 = "person, location, event, organization, concept"

print(f"\n列表格式: {entity_types1}")
print(f"字符串格式: {entity_types2}")

# 测试填充
test_prompt = prompt_template.format(
    entity_types=entity_types2,  # 使用字符串格式
    input_text="Test text",
    tuple_delimiter=prompts.PROMPTS['DEFAULT_TUPLE_DELIMITER'],
    record_delimiter=prompts.PROMPTS['DEFAULT_RECORD_DELIMITER'],
    completion_delimiter=prompts.PROMPTS['DEFAULT_COMPLETION_DELIMITER']
)

# 找到包含类型的行
for line in test_prompt.split('\n'):
    if 'types:' in line:
        print(f"\n填充后的类型行: {line}")
        break
