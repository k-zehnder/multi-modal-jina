import os
from docarray import DocumentArray, Document
from docarray.array.sqlite import SqliteConfig


# ----------- Init and add doc to new db
cfg = SqliteConfig(connection='imgs.db', table_name='da_all_table')
da = DocumentArray(storage='sqlite', config=cfg)
da.append(Document(text="some_doc"))
# da.summary()
print(da)
print(da[0].text)

# ----------- Read from newly init db
cfg = SqliteConfig(connection='existing.db', table_name='sample_table')
da = DocumentArray(storage='sqlite', config=cfg)
# da.summary()
print(da)
print(da[0].text)
# documents on disk

da_all = DocumentArray.from_files(['./tattoo_full/**/*.png', './tattoo_full/**/*.jpg', './tattoo_full/**/*.jpeg'])
print(da_all)
