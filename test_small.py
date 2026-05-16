#!/usr/bin/env python3
"""简单测试LLM - 使用phi4-mini模型"""

import ollama

print("测试Ollama LLM (使用phi4-mini)...")

try:
    response = ollama.generate(
        model="phi4-mini:3.8b",
        prompt="Hello! Who is Duan Yu?",
        options={'temperature': 0.1, 'num_predict': 500}
    )
    
    print("LLM响应:")
    print(response['response'])
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
