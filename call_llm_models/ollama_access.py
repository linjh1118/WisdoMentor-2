from utils.CallOllama import CallOllama

model_name = 'gemma:2b'

def call_ollama(prompt):
    ans = CallOllama.generate(model_name=model_name,text=prompt)
    return ans
