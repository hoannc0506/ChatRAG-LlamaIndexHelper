import logging 
import sys 
import nest_asyncio
from llama_index.core import (
    StorageContext, Settings, VectorStoreIndex
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
import model_utils, agent_utils, index_utils

# async config
nest_asyncio.apply()

# logging config
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout)) 

def main(device="cuda:0"):
    # load llm and embedding
    Settings.llm, Settings.embed_model = agent_utils.load_llm_embed_models(
        llm_name="models/Meta-Llama-3.1-8B-Instruct", 
        embed_name="models/bge-base-en-v1.5",
        temperature=0.1,
        device_map=device
    )
        
    # load vector index, enable hybrid search
    vector_index = index_utils.creat_qdrant_index(
        qdrant_dir='qdrant_db/',
        col_name='md-llama-blogs-hybrid',
        hybrid_search=True,
        fastembed_sparse_model='Qdrant/bm42-all-minilm-l6-v2-attentions'
    )

    # init re-ranker
    rerank_postprocessor = SentenceTransformerRerank(
        model='models/mxbai-rerank-xsmall-v1',
        top_n=5, # number of nodes after re-ranking,
        keep_retrieval_score=True,
        device=device
    )
    
    # querying
    query_engine = vector_index.as_query_engine(
        response_mode="compact",
        vector_store_query_mode="hybrid",
        node_postprocessors=[rerank_postprocessor],
        similarity_top_k=10, # semantic search nodes
        sparse_top_k=12, # hybrid search nodes
        use_async=True,
        streaming=True
    )

    # enable hyde
    hyde = HyDEQueryTransform(include_original=True)
    hyde_query_engine = TransformQueryEngine(query_engine, hyde)

    # sample questions
    questions = [
    "What are key features of llama-agents?",
    '''What are the two critical areas of RAG system performance that are \
assessed in the "Evaluating RAG with LlamaIndex" section of the OpenAI Cookbook?'''
]
    for idx, question in enumerate(questions):
        print("Question:", question)
        response = hyde_query_engine.query(question)
        response.print_response_stream()
        # print_ref_docs(response.metadata)
        print("\n","=="*40)
    

if __name__ == "__main__":
    import fire
    fire.Fire(main)