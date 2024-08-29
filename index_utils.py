# logging setup
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from llama_index.core import (
    SimpleDirectoryReader, Settings, StorageContext, 
    VectorStoreIndex, QueryBundle
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
import json
import chromadb
import qdrant_client
import data_utils

def check_exist_coll(db_dir, db_type):
    pass
    

def create_chroma_index(
    db_dir="./chroma_db", coll_name="md-llama-blogs",
    docs_dir="./data/llama-blogs-md", docs_metadata="data/llama_blogs_metadata.json",
    embed_model="models/bge-base-en-v1.5", device_map="cuda:1", re_indexing=False
):
    chroma_client = chromadb.PersistentClient(path=db_dir)
    
    # check exist collection
    colls = chroma_client.list_collections()
    coll_names = [coll.name for coll in colls]

    # Init chromadb
    vector_collection = chroma_client.get_or_create_collection(coll_name)

    # Init chromadb storage
    vector_store = ChromaVectorStore(
        chroma_collection=vector_collection,
        persist_dir=db_dir
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    if coll_name in coll_names and not reindexing:
        print("Loading index from vector store")
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            show_progress=True
        )
    else:
        print("Creating index from documents")
        if Settings.embed_model is None:
            # load embeddings
            Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model, device=device_map)
    
        # load nodes from documents
        nodes = data_utils.load_md_documents(
            docs_dir=docs_dir, docs_metadata=docs_metadata, return_nodes=True
        )

        # create vector index
        vector_index = VectorStoreIndex(
            nodes,
            storage_context=storage_context, 
            show_progress=True
        )

    return vector_index
    

def creat_qdrant_index(
    qdrant_dir="qdrant_db/", col_name="md-llama-blogs",
    embed_model="models/bge-base-en-v1.5", device_map="cuda:0",
    docs_dir="data/llama-blogs-md", docs_metadata="data/llama_blogs_metadata.json",
    hybrid_search=False, fastembed_sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions"
):

    # init qdrant db
    client = qdrant_client.QdrantClient(
        path=qdrant_dir
    )

    if hybrid_search:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=col_name,
            enable_hybrid=True,
            fastembed_sparse_model=fastembed_sparse_model
        )
    else:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=col_name,
        )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if client.collection_exists(col_name):
        print("Creating index from vector store")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            show_progress=True,
        )
        
    else:
        print("Creating index from documents store")
        # load embeddings
        Settings.llm = None
        if Settings.embed_model is None:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embed_model, 
                device=device_map
            )
            
        # load nodes from documents
        nodes = data_utils.load_md_documents(
            docs_dir=docs_dir, docs_metadata=docs_metadata, return_nodes=True
        )
        
        print(f"Loaded {len(nodes)} documents")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=True,
        )
        
    return index

def test_vector_index(db_type, device_map="cuda:1"):
    assert db_type in ['qdrant', 'chroma']
    
    # embed models
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="models/bge-base-en-v1.5",
        device=device_map
    )
    Settings.llm = None
    
    # indexing
    if db_type == 'qdrant':
        vector_index = creat_qdrant_index(device_map=device_map)
    else: 
        vector_index = create_chroma_index(device_map=device_map)
    
    query = '''What are the two critical areas of RAG system performance that are assessed \
in the "Evaluating RAG with LlamaIndex" section of the OpenAI Cookbook?'''
    query_bundle = QueryBundle(query)

    retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=5,
    )

    retrieved_nodes = retriever.retrieve(query_bundle)
    print("Question:", query)
    print("Retrieved nodes:")
    for idx, node in enumerate(retrieved_nodes):
        print(f"Node {idx}\n", json.dumps(node.metadata, indent=2))
        print("=="*40)

if __name__ == "__main__":
    import fire
    fire.Fire(test_vector_index)