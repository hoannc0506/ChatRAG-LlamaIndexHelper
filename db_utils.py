import nest_asyncio
import logging
import sys
import os
from tqdm import tqdm
from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, Settings
from llama_index.readers.file import HTMLTagReader, FlatReader
from llama_index.core.node_parser import HTMLNodeParser, MarkdownNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate
import qdrant_client
import model_utils
import prompt_utils
import fire

# async config
nest_asyncio.apply()

# logging config
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

def load_md_documents(docs_dir, docs_metadata, return_nodes=False):
    md_reader = FlatReader()
    parser = MarkdownNodeParser()
    
    docs_metadata = json.load(open("docs_metadata", "r"))
    print("Num documents:", len(docs_metadata))

    loaded_documents = []
    for doc_metadata in tqdm(docs_metadata, desc="Parsing documents"):
        file_path = os.path.join(docs_dir, doc_metadata["url"].split("/")[-1] + ".md")
        md_docs = md_reader.load_data(Path(file_path), extra_info=doc_metadata)
        loaded_documents = loaded_documents + md_docs 

    
    if return_nodes:
        nodes = parser.get_nodes_from_documents(loaded_documents, show_progress=True)
        return nodes

    return load_md_documents


def parse_md_doc(doc):
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(loaded_documents, show_progress=True)
    
    return nodes

    
def load_qdrant_db(local_path, coll_name):
    client = qdrant_client.QdrantClient(path=local_path)
    vector_store = QdrantVectorStore(client=client, collection_name=coll_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return vector_store, storage_context
    