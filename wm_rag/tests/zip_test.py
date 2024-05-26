import sys

sys.path.append(f"{sys.path[0]}/..")

from langchain_core.documents import Document

from prompt_zip.LongLlmLingua import LongLlmLinguaZipper
from prompt_zip.LlmLingua2 import LlmLingua2Zipper

test_str = [
    "微生物的个体极其微小，必须借助光学显微镜或电子显微镜才能观察到它们。测量其大小通常以微米(μm)或纳米(nm)为单位，微生物本身具有极为巨大的比表面积，小体积大面积必然有一个巨大的营养物质的吸收面，代谢废物的排泄面和环境信息的接触面，这对于微生物与环境之间进行物质、能量和信息的交换极为有利。当然也有体积较大的微生物存在，如担子菌等大型真菌，其子实体较大。",
    "微生物代谢旺盛，主要表现在吸收营养物质多，物质转化快这两个方面。大肠杆菌每小时可消耗达自身重量2000倍的糖类，乳酸细菌每小时吸收的营养物质达自身重量的100多倍，人类每小时吸收营养物质的量不及自身重量的0.3%。乳酸细菌每小时可产生达自身重量1000倍的乳酸，产原假丝酵母(candidautilis)合成蛋白质的能力是大豆的100倍，是肉用公牛的1000000倍。这些特性为微生物的高速生长繁殖和合成大量代谢产物提供了充分的物质基础，也使微生物获得了“活的化工厂”的美名。",
    "微生物的代谢类型多样，这是其他生物不可比拟的。微生物能利用的营养基质十分广泛，几乎能分解地球上的一切有机物质，许多动植物不能利用甚至对其他生物有毒的物质，微生物也可以利用。微生物有多种产能方式，有的可以分解有机物获能，有的可以氧化无机物获能，有的能利用光能进行光合作用，有的能固定分子态氮，有的能利用复杂有机氮化物。微生物的代谢产物更是多种多样的，氨基酸、蛋白质、糖类、核苷酸、核酸、脂肪、脂肪酸、抗生素、维生素、色素、生物碱、二氧化碳、H₂O、H₂S等都可以是微生物的代谢产物，仅抗生素就已发现9000多种。",
]

docs = [Document(page_content=content) for content in test_str]

zipper1 = LongLlmLinguaZipper(
    router_path="http://localhost", port="10003", path="longllmlingua"
)
zipper2 = LlmLingua2Zipper(
    router_path="http://localhost", port="10003", path="llmlingua2"
)

res1 = zipper1.prompt_zip(docs, "什么是微生物？")
res2 = zipper2.prompt_zip(docs, "什么是微生物？")

print("long llm lingua:", res1)
print("llm lingua2:", res2)
