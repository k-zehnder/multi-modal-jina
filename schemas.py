from typing import List
from pydantic import BaseModel
from docarray import Document, DocumentArray
from docarray.document.pydantic_model import PydanticDocument, PydanticDocumentArray


# {
#   "text": "name_default",
#   "uri": "./docs/usage/stars.jpg"
# }

# {
#     "text" : "name_default",
#     "uris" : ["./docs/usage/stars.jpg"]
# }


class Item(PydanticDocument):
    text: str = "name_default"
    uri: str = "./docs/usage/stars.jpg"

class MultipleItems(PydanticDocumentArray):
    text: str = "name_default"
    uris: List[str] = ["./docs/usage/stars.jpg"]

item1 = Item()