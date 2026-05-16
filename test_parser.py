#!/usr/bin/env python3
"""测试nano-graphrag的解析"""

import asyncio
import ollama
import nano_graphrag.prompt as prompts

async def test():
    """测试实体解析"""
    test_text = "李白在洛阳遇到了杜甫。杜甫是唐代诗人，担任左拾遗。"
    
    # 使用正确的分隔符
    prompt_template = prompts.PROMPTS['entity_extraction']
    filled_prompt = prompt_template.format(
        entity_types="person, location, event, organization, concept",
        input_text=test_text,
        tuple_delimiter=prompts.PROMPTS['DEFAULT_TUPLE_DELIMITER'],
        record_delimiter=prompts.PROMPTS['DEFAULT_RECORD_DELIMITER'],
        completion_delimiter=prompts.PROMPTS['DEFAULT_COMPLETION_DELIMITER']
    )
    
    print("发送请求到 qwen3.5...")
    response = ollama.chat(
        model="qwen3.5:9b",
        messages=[
            {
                'role': 'user',
                'content': filled_prompt
            }
        ],
        options={
            'temperature': 0.1,
            'num_predict': 1500,
            'think': False,
        }
    )
    
    result = response['message']['content']
    
    print(f"\n原始输出:")
    print("-" * 70)
    print(result if result else "(空)")
    print("-" * 70)
    
    print(f"\n输出长度: {len(result) if result else 0} 字符")
    print(f"是否包含<|COMPLETE|>: {'<|COMPLETE|>' in result if result else 'N/A'}")

if __name__ == "__main__":
    asyncio.run(test())
