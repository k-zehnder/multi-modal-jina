from jina import DocumentArray, Document
from jina.clients import Client
from jina import Executor, Flow, requests
from jina.types.request import Request
from docarray import DocumentArray, Document
from docarray.array.sqlite import SqliteConfig
from helpers import create_query_da, get_embedded_da_from_img_files, plot_search_results, get_client, search_by_text, show_results, show_montage


if __name__ == "__main__":
    # NOTE: need to have app.py backend running before this will work

    # ---------- SETUP CLIENT
    c = get_client()

    # ---------- PERFORM SEARCH
    # query_text = "gambling"
    # query_text = "flowers"
    # query_text = "fish"
    # query_text = "stars"
    # query_text = "guns"
    # query_text = "army soldier"
    # query_text = "video games"
    # query_text = "hamburgers"
    # query_text = "marijuana"
    # query_text = "spiders"
    # query_text = "rock and roll"
    # query_text = "airplane"
    # query_text = "lion"
    query_text = "butterfly"

    results = search_by_text(c, query_text, verbose=True)
    
    show_montage(query_text, results)
    import cv2
    cv2.destroyAllWindows()
