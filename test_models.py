#!/usr/bin/env python3
"""测试不同的模型"""

import ollama

def test_model(model_name):
    """测试单个模型"""
    print(f"\n测试模型: {model_name}")
    print("-" * 70)
    
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                'role': 'user',
                'content': '你好，请介绍一下李白'
            }
        ]
    )
    
    content = response['message']['content']
    print(f"结果长度: {len(content)} 字符")
    print(f"前200字符: {content[:200] if content else '(空)'}")
    return len(content) > 0

if __name__ == "__main__":
    models = ['qwen2', 'qwen3.5:9b', 'phi4-mini:3.8b', 'llama3:8b']
    
    for model in models:
        try:
            success = test_model(model)
            if success:
                print("✅ 成功")
            else:
                print("❌ 返回空")
        except Exception as e:
            print(f"❌ 错误: {e}")
