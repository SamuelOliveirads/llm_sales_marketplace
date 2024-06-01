from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def update_chroma_db() -> Chroma:
    chroma_db_dir = "data/03_primary/chroma_db"
    docsearch = Chroma(
        persist_directory=chroma_db_dir,
        embedding_function=OpenAIEmbeddings(model="text-embedding-ada-002"),
    )

    docsearch.persist()
    retriever = docsearch.as_retriever()

    return retriever
