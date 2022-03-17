import os
from docarray import DocumentArray, Document
from docarray.array.sqlite import SqliteConfig

# ----------- Init and add doc to new db
cfg = SqliteConfig(connection='example.db', table_name='table1')
da = DocumentArray(storage='sqlite', config=cfg)
da.append(Document(text="some_doc"))
# da.summary()
print(da)
print(da[0].text)

# ----------- Read from newly init db
cfg = SqliteConfig(connection='example.db', table_name='table1')
da = DocumentArray(storage='sqlite', config=cfg)
# da.summary()
print(da)
print(da[0].text)