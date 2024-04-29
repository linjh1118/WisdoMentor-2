# Kimi Unofficial API

## 使用

1. 激活环境

  ``` shell
  conda activate kimi
  ```

  或者安装依赖

  ``` shell
  pip install -r requirements.txt
  ```

2. 运行main.py

  - 参数：
    - -h, --help 显示帮助信息
    - -u, --upload-file 指定上传文件目录
    - -q, --question-file 指定问题文件所在目录
    - -o, --output-file 指定输出文件目录
    - -t, --test-file 是否使用测试文件

  - 示例：

  ``` shell
    python main.py -u /home/chy/dream/hjc/kimi/test/uploads -q /home/chy/dream/hjc/kimi/test/questions -o /home/chy/dream/hjc/kimi/test/outputs
    python "C:\Users\cgj\Desktop\Python项目\rag\kimi(1)\kimi\main.py" -u "D:\paper" -q "C:\Users\cgj\Desktop\Python项目\rag\kimi(1)\kimi\QAs\2207.07051_out.txt" -o "C:\Users\cgj\Desktop\Python项目\rag\kimi(1)\kimi\test\outputs"
  ```

  也可以使用测试文件进行测试：

  ``` shell
  python main -t
  ```

## 目录结构说明 

  - 上传文件目录：仅包含上传文件，如：

    ``` shell
    |- .
    |- ..
    |- file1.pdf
    |- file2.pdf
    |- file3.txt
    |- file4.docx
    |- ...
    ```
  
  - 问题文件目录：与上传文件同名，内容为若干个问题，每个问题一行，如：

    ``` shell
    |- .
    |- ..
    |- file1.txt
    |- file2.txt
    |- file3.txt
    |- file4.txt
    |- ...
    ```

  - 输出文件目录：若不提供，则默认输出至上传文件目录，如：
  
    ``` shell
    |- .
    |- ..
    |- file1_out.txt
    |- file2_out.txt
    |- file3_out.txt
    |- file4_out.txt
    |- ...
    ```

    由于Kimi的回复可能包含多行，故将问题以`Q:`开头，回复以`A:`开头，如：
      
    ``` shell
    Q:这篇文章的主要内容是什么？
    A:这篇论文探讨了预训练语言模型（PLMs）中的彩票提示（lottery prompts）问题，并研究了这些提示的泛化能力。彩票提示是指对于PLMs中的每个实例，几乎总能找到至少一个能够诱导模型产生正确预测的提示。研究者们首先验证了对于任何给定的分类任务中的每个实例，都存在至少一个彩票提示，并且这些提示可以通过自动搜索过程以较低的成本获得。此外，他们发现一些强提示（strong prompts）在整个训练集上表现出色，并且具有可区分的语言特征。最后，研究者们尝试使用提示集成方法将搜索到的强提示泛化到未见过的数据上，而不进行任何参数调整。

    研究者们在多种NLP分类任务上进行了实验，证明了所提出的方法能够与无需梯度和优化的其他基线方法取得可比的结果。他们还对彩票提示的搜索成本和搜索到的彩票提示进行了深入分析，发现搜索成本与数据难度和模型容量有关。此外，他们还探讨了强提示的语言特性，并发现这些提示具有依赖于标签词和任务类型的明显语言特征。

    论文还讨论了相关工作，包括提示工程和自动提示生成的最新进展，并提出了未来研究的方向，例如如何更有效地使用PLM推理调用以及如何缩小小型PLM和大型语言模型之间的差距。最后，论文指出了研究的局限性和潜在的伦理考虑，强调了合理使用PLMs的重要性，并提出了未来研究的方向，以便更有效地挖掘和利用彩票提示。
    ```

## 代码结构说明

  ``` shell
  |- .
  |- ..
  |- __pycache__
  |- libs # 存放处理对话、文件、token的代码
    |- .
    |- ..
    |- __init__.py
    |- consts.py # 存放常量
    |- conversation.py # 处理对话
    |- file.py # 处理文件
    |- utils.py # 处理token与检测请求是否正常被回复
    |- refresh_token # 存放refresh_token
  |- test
    |- .
    |- ..
    |- outputs # 测试输出
    |- questions # 测试问题文件
    |- uploads # 测试上传文件
  |- main.py # 主程序
  |- README.md # 本文档
  |- requirements.txt # 依赖
  |- backup.zip # 备份文件
  ```
