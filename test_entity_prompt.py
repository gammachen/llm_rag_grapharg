#!/usr/bin/env python3
"""测试qwen3.5对实体提取提示词的反应"""

import asyncio
import ollama
import nano_graphrag.prompt as prompts

async def test():
    """测试"""
    test_text = """
    In 744 AD, Li Bai met Du Fu in Luoyang. Du Fu was a great Tang Dynasty poet 
    who later served as "Zuoshiyi", a low-ranking official position. 
    They became close friends and traveled together.
    """
    
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

asyncio.run(test())
