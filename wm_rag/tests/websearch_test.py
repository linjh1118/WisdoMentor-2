import sys

sys.path.append(f"{sys.path[0]}/..")

from websearch import BaiduCallback
from websearch import ArxivCallback

# from web_extract import TextExtractor
# from web_extract import PdfExtractor
# from web_extract import KimiExtractor

query = "computer science"

baidu_searcher = BaiduCallback()
arxiv_searcher = ArxivCallback()

baidu_res = baidu_searcher.search(query)
arxiv_res = arxiv_searcher.search(query)

print(baidu_res)
print(arxiv_res)

# text_extractor = TextExtractor()
# pdf_extractor = PdfExtractor()

# text_res = text_extractor.extract(baidu_res)
# pdf_res = pdf_extractor.extract(arxiv_res)

# text_res = [doc.page_content for doc in text_res if doc.page_content]
# pdf_res = [doc.page_content for doc in pdf_res if doc.page_content]

# print(len(text_res))
# print(len(pdf_res))

# kimi_baidu_res = baidu_searcher.search(query, max_res=1)
# kimi_arxiv_res = arxiv_searcher.search(query, max_res=1)

# kimi_extractor = KimiExtractor()

# kimi_baidu_res = kimi_extractor.extract(kimi_baidu_res)
# kimi_arxiv_res = kimi_extractor.extract(kimi_arxiv_res)

# print(kimi_baidu_res)
# print(kimi_arxiv_res)
