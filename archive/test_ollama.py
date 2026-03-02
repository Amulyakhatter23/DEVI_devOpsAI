import ollama
import os

print(f"OLLAMA_HOST env var: {os.environ.get('OLLAMA_HOST')}")

try:
    print("Attempting chat with options...")
    response = ollama.chat(
        model='qwen2.5-coder:7b', 
        options={"temperature": 0.0},
        messages=[{'role': 'user', 'content': 'Hello'}]
    )
    print("Chat success!")
    print(response['message']['content'])
    
except Exception as e:
    print(f"\n❌ Error: {e}")
