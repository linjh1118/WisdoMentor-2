## 平替Chatgpt的使用指南


**将目标代码中和chatgpt交互的代码，改成如下代码。**

```
# test.py
from utils.CallOllama import CallOllama
question = '吉林大学在哪？'
ans = CallOllama.generate(model_name = 'gemma:2b', text = question)
```

- 正常情况下，执行上边代码，终端会打印出如下内容：
    ```
    (llama_factory) chy@chy-desktop:~/dream/linjh$ python test.py 
    #############################################
    ###### input to gemma:2b in Ollama ######
    吉林大学在哪？
    ###### response of gemma:2b in Ollama ######
    吉林大学是位于中国吉林省的国家级大学，成立于1958年。
    函数 generate 的执行时间为：8.37109923362732 秒
    ```

- 使用前，确保环境中有ollama-python，如果没有，可以使用以下命令安装：
    ```
    chy@chy-desktop:~/dream/linjh$ pip install ollama
    ....
    ....

    # 供参考
    # 目前llama_factory中的ollama-python版本
    (llama_factory) chy@chy-desktop:~/dream/linjh$ pip show ollama
    Name: ollama
    Version: 0.1.8
    ```
----