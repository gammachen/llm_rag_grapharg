#!/usr/bin/env python3
"""查看nano-graphrag的提示词"""

import nano_graphrag.prompt as prompts

# 查看实体提取提示词
prompt_template = prompts.PROMPTS['entity_extraction']

test_text = "李白在洛阳遇到了杜甫。杜甫是唐代诗人，担任左拾遗。"

filled_prompt = prompt_template.format(
    entity_types="person, location, event, organization, concept",
    input_text=test_text,
    tuple_delimiter=prompts.PROMPTS['DEFAULT_TUPLE_DELIMITER'],
    record_delimiter=prompts.PROMPTS['DEFAULT_RECORD_DELIMITER'],
    completion_delimiter=prompts.PROMPTS['DEFAULT_COMPLETION_DELIMITER']
)

print("=" * 70)
print("实体提取提示词")
print("=" * 70)
print(filled_prompt)
print("=" * 70)
