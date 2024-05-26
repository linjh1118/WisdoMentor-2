FROM nvidia/cuda:12.1.0-base-ubuntu20.04

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
  sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get install -y --no-install-recommends python3 python3-pip && \
  rm -rf /var/lib/apt/lists/* && \
  python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app
COPY app /app

RUN pip3 install -r requirements.txt

RUN HF_ENDPOINT=https://hf-mirror.com  huggingface-cli download --resume-download shenzhi-wang/Llama3-8B-Chinese-Chat --local-dir weights/Llama-3-8b \
  && HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download microsoft/llmlingua-2-xlm-roberta-large-meetingbank --local-dir weights/llmlingua-2-xlm-roberta-large

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10003"]