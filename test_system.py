#!/usr/bin/env python3
"""测试添加系统提示词"""

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
    
    print("测试: ollama.chat + 系统提示词")
    print("-" * 70)
    response = ollama.chat(
        model="qwen3.5:9b",
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant. You must respond directly without using <think> tags.'
            },
            {
                'role': 'user',
                'content': filled_prompt
            }
        ],
        options={
            'temperature': 0.1,
            'num_predict': 1500,
        }
    )
    content = response['message']['content']
    print(f"结果长度: {len(content)} 字符")
    print(f"前300字符:\n{content[:300] if content else '(空)'}")
    
    print("\n\n测试2: 简短提示词")
    print("-" * 70)
    short_prompt = "Extract entities from this text: Li Bai met Du Fu in Luoyang. Output in format with <|COMPLETE|> at end."
    response2 = ollama.chat(
        model="qwen3.5:9b",
        messages=[
            {
                'role': 'user',
                'content': short_prompt
            }
        ],
        options={
            'temperature': 0.1,
            'num_predict': 500,
        }
    )
    content2 = response2['message']['content']
    print(f"结果长度: {len(content2)} 字符")
    print(f"内容:\n{content2 if content2 else '(空)'}")

if __name__ == "__main__":
    test()
