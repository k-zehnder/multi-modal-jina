from helpers import get_docs_from_sqlite, get_embedded_da_from_img_files
from docarray import Document, DocumentArray


if __name__ == "__main__":
    connection = "./workspace/SimpleIndexer/0/tattoo_images_index.db"
    table = "clip"
    da = get_docs_from_sqlite(connection, table) # THESE ARE FEATURE VECTORS NOT IMAGES!!
    q = Document(uri="/Users/peppermint/Desktop/codes/python/multi-modal-jina/data/tattoo_images/Tattoo_Butterfly2.JPG")
    print(da)
    print(q)

    results = q.match(da)
    print(results)
    # for m in results.matches:
    #     print(q.uri, m.uri, m.scores['cosine'].value)