#!/usr/bin/env python3
import asyncio
import ollama

async def test():
    response = await ollama.chat(
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
    print("结果:", response)
    print("类型:", type(response))
    print("内容:", response['message']['content'][:200] if response['message']['content'] else "(空)")

if __name__ == "__main__":
    asyncio.run(test())
