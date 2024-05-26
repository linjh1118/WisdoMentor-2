FROM python:3.11.0

RUN python3 -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
  pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app
COPY app /app

RUN pip3 install -r requirements.txt

EXPOSE 10001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10001"]
