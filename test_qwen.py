#!/usr/bin/env python3
"""测试qwen3.5的实体提取"""

import asyncio
import ollama
import nano_graphrag.prompt as prompts

async def test_qwen():
    """测试qwen3.5模型"""
    test_text = "李白在洛阳遇到了杜甫。杜甫担任左拾遗。"
    
    prompt_template = prompts.PROMPTS['entity_extraction']
    filled_prompt = prompt_template.format(
        entity_types="person, location, event, organization, concept",
        input_text=test_text,
        tuple_delimiter="<|>|",
        record_delimiter="\\n",
        completion_delimiter="<|COMPLETE|>"
    )
    
    print("发送请求到 qwen3.5（禁用思考）...")
    response = ollama.generate(
        model="qwen3.5:9b",
        prompt=filled_prompt,
        options={
            'temperature': 0.1,
            'num_predict': 1500,
            'think': False,  # 禁用思考模式
        }
    )
    
    result = response['response']
    
    print(f"\n原始输出:")
    print("-" * 70)
    print(result if result else "(空)")
    print("-" * 70)
    
    print(f"\n输出长度: {len(result)} 字符")
    print(f"是否包含<|COMPLETE|>: {'<|COMPLETE|>' in result}")

asyncio.run(test_qwen())
