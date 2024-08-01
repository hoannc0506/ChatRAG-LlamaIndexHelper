import nest_asyncio
import logging
import sys

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import StorageContext, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts import PromptTemplate
import qdrant_client
import model_utils
import prompt_utils

# async config
nest_asyncio.apply()

# logging config
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def load_qdrant_db(local_path, coll_name):
    client = qdrant_client.QdrantClient(path=local_path)
    vector_store = QdrantVectorStore(client=client, collection_name=coll_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    return vector_store, storage_context
    