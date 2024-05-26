FROM ollama/ollama:0.1.39

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
  sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
  apt-get update && apt-get -y install wget curl aria2 git git-lfs && \
  wget https://hf-mirror.com/hfd/hfd.sh && chmod a+x hfd.sh && \
  HF_ENDPOINT=https://hf-mirror.com ./hfd.sh shenzhi-wang/Llama3-8B-Chinese-Chat-GGUF-8bit --tool aria2c -x 4 --local-dir /weights/llama3-8b-chinese

COPY Modelfile /weights/llama3-8b-chinese/Modelfile

RUN ollama serve & sleep 10 && \
  ollama create llama3-8b-chinese -f /weights/llama3-8b-chinese/Modelfile && \
  rm -rf /weights/llama3-8b-chinese

ENTRYPOINT ["/bin/ollama"]

CMD ["serve"]