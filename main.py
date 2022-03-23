from docarray.document.pydantic_model import PydanticDocument, PydanticDocumentArray
from fastapi import FastAPI
from pydantic import BaseModel
from docarray import Document, DocumentArray
import cv2
import numpy as np
from typing import List
from docarray.array.sqlite import SqliteConfig
from helpers import get_client, search_by_text, show_results, get_docs_from_sqlite
from schemas import Item, MultipleItems


app = FastAPI()


class Query(BaseModel):
    query_text: str = "video games"

@app.post('/get_match')
def get_match(query: Query):
    # NOTE: need to have create_data.py running in background for this to work
    c = get_client()
    query_text = query.query_text
    results = search_by_text(c, query_text, verbose=True)
    results = [[query_text, m.uri, m.scores['cosine'].value] for m in results[0].matches]
    print(f'results: {results}')
    return {"results" : results}

@app.post('/single')
async def create_item(item: PydanticDocument):
    d = Document.from_pydantic_model(item)
    return d.to_pydantic_model()

@app.post('/multiple')
async def create_items(multiple_items: PydanticDocumentArray):
    da = DocumentArray.from_pydantic_model(multiple_items)
    img = cv2.imread(da[0].uri)
    print(img)
    # now `d` is a Document object
    # process `d` how ever you want
    return da.to_pydantic_model()

# @app.post('/get_match_db')
# def get_match(query: Query):
#     connection = "./workspace/SimpleIndexer/0/tattoo_images_index.db"
#     table = "clip"
#     da = get_docs_from_sqlite(connection, table)
#     return da
