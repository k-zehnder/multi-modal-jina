import os
import cv2
import imutils
from imutils import paths
from docarray import DocumentArray, Document
from docarray.array.sqlite import SqliteConfig
from jina import DocumentArray, Document
from jina.clients import Client
from jina import Executor, Flow, requests
from jina.types.request import Request
from docarray import DocumentArray, Document
from docarray.array.sqlite import SqliteConfig

from resultsmontage import ResultsMontage


def get_docs_from_sqlite(connection, table):
    cfg = SqliteConfig(connection, table)
    return DocumentArray(storage='sqlite', config=cfg)

def get_embedded_da_from_img_files(images_path, num):
    return DocumentArray.from_files(images_path, num).apply(
        lambda d: d.load_uri_to_image_tensor()
        .load_uri_to_image_tensor(200, 200)  # load
        .set_image_tensor_normalization()  # normalize color
        .set_image_tensor_channel_axis(-1, 0)  # switch color axis for the PyTorch model later
    )    

def create_query_da(search_term):
    return DocumentArray(Document(text=search_term))

def plot_search_results(resp: Request):
    for doc in resp.docs:
        print(f'Query text: {doc.text}')
        print(f'Matches:')
        print('-' * 10)
        # doc.matches[:3].plot_image_sprites()
        print([d.uri for d in doc.matches[:3]])
        print()

def get_client(port=12345, show_progress=True):
    c = Client(port=port)
    c.show_progress = show_progress
    return c

def search_by_text(client, query_text, verbose=False):
    q = create_query_da(query_text)
    results = client.post('/search', inputs=q, return_results=True)
    if verbose:
        show_results(q, results)
    return results

def show_results(query, results):
    print(f"query_text: {query[0].text}")
    for d in results:
        for m in d.matches:
            print(d.uri, m.uri, m.scores['cosine'].value)
    return results

def load_caltech(path):
    da = DocumentArray()
    imagePaths = paths.list_images(path)
    for imagePath in imagePaths:
        da.append(Document(uri=imagePath))
    da.apply(
            lambda d: d.load_uri_to_image_tensor()
            .load_uri_to_image_tensor(200, 200)  # load
            .set_image_tensor_normalization()  # normalize color
            .set_image_tensor_channel_axis(-1, 0)  # switch color axis for the PyTorch model later
        )    
    return da

def show_montage(query, res):
    print(f'query_text: {query}')

    montage = ResultsMontage((600, 800), 3, 12)
    for (i, match) in enumerate(res.to_dict()[0]["matches"]):
        result = cv2.imread(match["uri"]) 
        score = match['scores']['cosine']['value']
        
        montage.addResult(result, text=f"#{i+1} > {score:.2f}")

        cv2.imshow(f"Query Text: {query}", imutils.resize(montage.montage, height=800))
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    conn = "./workspace/SimpleIndexer/0/index.db"
    table = "clip"
    
    data = get_embedded_da_from_img_files(conn, table)
    print(data)
