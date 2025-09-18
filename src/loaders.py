from langchain_community.document_loaders import WebBaseLoader, PyPDFium2Loader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_documents():
    # Option 1: PDF
    loader = PyPDFium2Loader("data/recipes/recipes_book.pdf")
    documents = loader.load()

    return documents

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048,
        chunk_overlap=0
    )
    return splitter.split_documents(documents)
