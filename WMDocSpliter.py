class WMDocSpliter:
    def __init__(self):
        # self.doc = doc
        # self.doc_len = len(doc)
        # self.doc_pos = 0
        pass

    def split_file(self, input_file_path, output_prefix_str, chunk_size_bytes):
        with open(input_file_path, "r", encoding="UTF-8") as f:
            chunk_no = 1
            while True:
                chunk = f.read(chunk_size_bytes)
                if not chunk:
                    break
                with open(f"{output_prefix_str}/{chunk_no}.txt", "w") as output:
                    output.write(chunk)
                chunk_no += 1

    def sentence_spliter(self, input_file_path, output_dir):
        import nltk
        from nltk.tokenize import sent_tokenize
        import nltk.tokenize.punkt

        # 定义句子分割函数
        def split_file_by_sentences(filename, output_dir):
            # 确保nltk的punkt分词器模型已下载
            try:
                nltk.data.find("tokenizers/punkt")
            except Exception as e:
                print("download...")
                nltk.download("punkt")

            # 读取文件内容
            with open(filename, "r", encoding="utf-8") as file:
                content = file.read()

            # 分割句子
            sentences = sent_tokenize(content)

            # 将每个句子写入新的文件
            for i, sentence in enumerate(sentences):
                # 写入文件
                with open(
                    f"{output_dir}/sentence_{i}.txt", "w", encoding="utf-8"
                ) as outfile:
                    outfile.write(sentence + "\n")  # 添加换行符

        split_file_by_sentences(input_file_path, output_dir)

    def intent_spliter(self):
        return NotImplementedError
