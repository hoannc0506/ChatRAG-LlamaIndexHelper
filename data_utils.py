import nest_asyncio
import logging
import sys
import os
import json
from tqdm import tqdm
from pathlib import Path

from llama_index.core import StorageContext, Settings
from llama_index.readers.file import FlatReader
from llama_index.core.node_parser import MarkdownNodeParser, SemanticSplitterNodeParser
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client
import fire


def load_md_documents(
    docs_dir="./data/llama-blogs-md", 
    docs_metadata="./data/llama_blogs_metadata.json", 
    return_nodes=False
):
    md_reader = FlatReader()
    parser = MarkdownNodeParser()
    
    docs_metadata = json.load(open(docs_metadata, "r"))
    print("Num documents:", len(docs_metadata))

    loaded_documents = []
    for doc_metadata in tqdm(docs_metadata, desc="Parsing documents"):
        file_path = os.path.join(docs_dir, doc_metadata["url"].split("/")[-1] + ".md")
        md_docs = md_reader.load_data(Path(file_path), extra_info=doc_metadata)
        loaded_documents = loaded_documents + md_docs 

    
    if return_nodes:
        nodes = parser.get_nodes_from_documents(loaded_documents, show_progress=True)
        return nodes

    return loaded_documents


def parse_md_doc(doc):
    md_parser = MarkdownNodeParser()
    nodes = md_parser.get_nodes_from_documents(doc, show_progress=True)
    
    return nodes


def semantic_chunking(docs, embed_model="models/bge-base-en-v1.5", device_map="cuda:1"):

    if isinstance(embed_model, str): 
        embed_model = HuggingFaceEmbedding(
            model_name=embed_model,
            device=device_map
        )

    semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )

    # semantic chunking
    splitted_nodes = semantic_splitter.get_nodes_from_documents(docs, show_progress=True)

    return splitted_nodes
    
def load_qdrant_db(local_path, coll_name):
    client = qdrant_client.QdrantClient(path=local_path)
    vector_store = QdrantVectorStore(client=client, collection_name=coll_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return vector_store, storage_context
    