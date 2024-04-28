import sys
import ollama
import time


def calculate_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__} 的执行时间为：{execution_time} 秒")
        return result
    return wrapper

class CallOllama:

    @staticmethod
    @calculate_execution_time
    def generate(model_name, text):
        response = ollama.chat(model=model_name, messages=[
            {
                'role': 'user',
                'content': text,
            },
        ])
        print('#############################################', file=sys.stderr)
        print('###### input to {model_name} in Ollama ######', file=sys.stderr)
        print(text, file=sys.stderr)
        print('###### response of {model_name} in Ollama ######', file=sys.stderr)
        print(response['message']['content'], file=sys.stderr)
        
        return response['message']['content'], response