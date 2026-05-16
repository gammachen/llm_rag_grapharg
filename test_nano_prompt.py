#!/usr/bin/env python3
"""测试nano-graphrag提示词"""

import ollama
import nano_graphrag.prompt as prompts

def test():
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
    
    print(f"提示词长度: {len(filled_prompt)} 字符")
    
    print("\n测试: qwen3.5:9b + nano-graphrag提示词")
    print("-" * 70)
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
            'num_predict': 2000,
        }
    )
    content = response['message']['content']
    print(f"结果长度: {len(content)} 字符")
    print(f"前500字符:\n{content[:500] if content else '(空)'}")
    
    print("\n\n测试: phi4-mini:3.8b + nano-graphrag提示词")
    print("-" * 70)
    response2 = ollama.chat(
        model="phi4-mini:3.8b",
        messages=[
            {
                'role': 'user',
                'content': filled_prompt
            }
        ],
        options={
            'temperature': 0.1,
            'num_predict': 2000,
        }
    )
    content2 = response2['message']['content']
    print(f"结果长度: {len(content2)} 字符")
    print(f"前500字符:\n{content2[:500] if content2 else '(空)'}")

if __name__ == "__main__":
    test()
