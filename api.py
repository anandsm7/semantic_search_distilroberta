from typing import Optional

from fastapi import FastAPI
from fastapi import responses
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from FAQ_RoBERTa import FAQBuilder

fb = FAQBuilder()


class input(BaseModel):
    query: str
    count: int

app = FastAPI()


@app.post("/predict/")
async def create_item(input: input): 
    res = fb.get_FAQ(input.query)
    response = dict()
    response['query'] = list(res['query'].values)
    response['confidence'] = list(res['similarity'].values)
    print(response)
    return response