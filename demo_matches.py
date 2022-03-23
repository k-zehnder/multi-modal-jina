from helpers import get_docs_from_sqlite


if __name__ == "__main__":
    connection = "./workspace/SimpleIndexer/0/tattoo_images_index.db"
    table = "clip"
    da = get_docs_from_sqlite(connection, table)
    print(da)