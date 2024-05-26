FROM nvidia/cuda:12.1.0-base-ubuntu20.04

RUN sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.tuna.tsinghua.edu.cn/ubuntu/|g' /etc/apt/sources.list && \
  sed -i 's|http://security.ubuntu.com/ubuntu|https://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get install -y --no-install-recommends python3 python3-pip && \
  rm -rf /var/lib/apt/lists/* && \
  python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
  pip3 install torch torchvision torchaudio

WORKDIR /app
COPY app /app

RUN pip3 install -r requirements.txt

RUN HF_ENDPOINT=https://hf-mirror.com  huggingface-cli download --resume-download BAAI/bge-m3 --local-dir weights/bge-m3 \
  && HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download google-bert/bert-base-chinese --local-dir weights/bert-chinese

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
