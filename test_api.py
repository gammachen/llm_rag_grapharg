#!/usr/bin/env python3
"""测试不同的Ollama API"""

import ollama
import nano_graphrag.prompt as prompts

def test():
    """测试不同的API"""
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
    
    print("测试1: ollama.generate")
    print("-" * 70)
    response1 = ollama.generate(
        model="qwen3.5:9b",
        prompt=filled_prompt,
        options={
            'temperature': 0.1,
            'num_predict': 1500,
            'think': False,
        }
    )
    print(f"结果长度: {len(response1['response'])} 字符")
    print(f"前200字符: {response1['response'][:200] if response1['response'] else '(空)'}")
    print()
    
    print("测试2: ollama.chat")
    print("-" * 70)
    response2 = ollama.chat(
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
    print(f"结果类型: {type(response2)}")
    content = response2['message']['content']
    print(f"结果长度: {len(content)} 字符")
    print(f"前200字符: {content[:200] if content else '(空)'}")

if __name__ == "__main__":
    test()
