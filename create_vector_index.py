# logging setup
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# llama index ascyncio config
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import fire

def main(local_db_dir, collection_name):
    # # load embeddings
    embed_model = HuggingFaceEmbedding(model_name="models/bge-small-en-v1.5", device="cuda")

    # update settings
    Settings.embed_model = embed_model
    Settings.llm = None
    
    # load documents
    documents = SimpleDirectoryReader(
        input_dir="./data",
        filename_as_id=True,
    ).load_data()
    # import pdb;pdb.set_trace()
    print(f"Loaded {len(documents)} documents")

    splitter = SentenceSplitter(
        chunk_size=1024,
        chunk_overlap=200
    )

    # split documents to nodes
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Splitted {len(documents)} documents to {len(nodes)} nodes.")

    # Creates a persistent instance of Chroma that saves to disk
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get or create a collection with the given name and metadata.
    vector_collection = chroma_client.get_or_create_collection(collection_name)
    
    print(vector_collection, vector_collection.count())
    
    # Init chromadb storage
    vector_store = ChromaVectorStore(
        chroma_collection=vector_collection,
        persist_dir=local_db_dir
    )
    vector_storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # create vector index
    vector_index = VectorStoreIndex(
        nodes,
        storage_context=vector_storage_context, 
        show_progress=True
    )

    # save index to local
    vector_index.storage_context.persist(local_db_dir)

if __name__ == "__main__":
    fire.Fire(main)