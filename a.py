# def my_function(arg1, arg2, arg3="default"):
#     parameter_names = list(locals().keys())
#     print(parameter_names)
#     a = locals()
#     print(a)

# c = 1
# my_function("value1", "value2", arg3="value3")

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/chy/dream/LLMs/bge-large-zh-v1.5')
    parser.add_argument('--tokenizer_path', type=str, default='/home/chy/dream/LLMs/bge-large-zh-v1.5')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--port', type=int, default=8888)
    args = parser.parse_args()
    return args

import sys
b = sys.argv
print(b)

args = get_args()

b = vars(args)
print(b)