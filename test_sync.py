#!/usr/bin/env python3
import ollama

response = ollama.chat(
    model="qwen3.5:9b",
    messages=[
        {
            'role': 'user',
            'content': '你好，请介绍一下李白'
        }
    ],
    options={
        'temperature': 0.1,
        'think': False,
    }
)

print("结果类型:", type(response))
print("消息内容:", response['message']['content'][:200] if response['message']['content'] else "(空)")
