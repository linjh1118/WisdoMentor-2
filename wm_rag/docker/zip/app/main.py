import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from models.LongLlmLingua import LongLlmLingua
from models.LlmLingua2 import LlmLingua2

app = FastAPI()


class ZipRequest(BaseModel):
    context: List[str]
    question: str
    rate: float = 0.5
    instruction: str = ""


@app.post("/longllmlingua")
def longllmlingua(request: ZipRequest):
    zipper = LongLlmLingua(
        os.path.join(os.path.dirname(__file__), "weights", "Llama-3-8b")
    )
    context = request.context
    question = request.question
    rate = request.rate
    instruction = request.instruction
    res = zipper.zip(context, question, rate, instruction)
    return {"compressed_prompt": res}


@app.post("/llmlingua2")
def llmlingua2(request: ZipRequest):
    zipper = LlmLingua2(
        os.path.join(
            os.path.dirname(__file__), "weights", "llmlingua-2-xlm-roberta-large"
        )
    )
    context = request.context
    context = request.context
    question = request.question
    rate = request.rate
    instruction = request.instruction
    res = zipper.zip(context, question, rate, instruction)
    return {"compressed_prompt": res}
