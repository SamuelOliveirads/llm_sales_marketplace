import pickle
from typing import List

from dotenv import load_dotenv
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()


def _save_to_file(obj: any, filename: str) -> None:
    """Save an object to a file using pickle.

    Parameters
    ----------
    obj : any
        The object to be saved.
    filename : str
        The filename where the object will be saved.
    """
    with open(filename, "wb") as file:
        pickle.dump(obj, file)


def _load_from_file(filename: str) -> any:
    """Load an object from a file using pickle.

    Parameters
    ----------
    filename : str
        The filename where the object is saved.

    Returns
    -------
    any
        The object loaded from the file.
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


def _load_text() -> List[LangchainDocument]:
    """Load and process text files.

    Returns
    -------
    List[LangchainDocument]
        The updated list of documents including text files.
    """
    filepath = "data/01_raw/products.txt"
    with open(filepath, "r", encoding="utf-8") as file:
        lines = file.readlines()

    documents = []
    for line in lines:
        parts = line.split(":")
        category = parts[0].strip()
        product_info = parts[1].split("-")
        product_name = product_info[0].strip()
        price = product_info[1].strip()

        metadata = {"category": category, "product_name": product_name, "price": price}
        doc = LangchainDocument(page_content=line.strip(), metadata=metadata)
        documents.append(doc)

    return documents


def load_data():
    docs = _load_text()

    compress_documents_files = "data/03_primary/compress_documents.pkl"

    _save_to_file(docs, compress_documents_files)

    docs = _load_from_file(compress_documents_files)

    chroma_db_dir = "data/03_primary/chroma_db"
    docsearch = Chroma.from_documents(
        docs,
        OpenAIEmbeddings(model="text-embedding-ada-002"),
        persist_directory=chroma_db_dir,
    )

    return docs, docsearch


if __name__ == "__main__":
    load_data()
