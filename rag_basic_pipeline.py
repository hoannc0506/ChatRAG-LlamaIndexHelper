import logging 
import sys 
import nest_asyncio
from llama_index.core import (
    StorageContext, Settings, VectorStoreIndex
)
import model_utils, agent_utils, index_utils

# async config
nest_asyncio.apply()

# logging config
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout)) 

def main():
    device_map = "cuda:0"
    # load llm and embedding
    Settings.llm, Settings.embed_model = agent_utils.load_llm_embed_models(
        llm_name="models/Meta-Llama-3.1-8B-Instruct", 
        embed_name="models/bge-base-en-v1.5", 
        device_map=device_map
    )
        
    # indexing
    vector_index = index_utils.creat_qdrant_index()
    
    # querying
    query_engine = vector_index.as_query_engine(
        response_mode="compact",
        use_async=True,
        streaming=True
    )

    # sample questions
    questions = [
    "What are key features of llama-agents?",
    '''What are the two critical areas of RAG system performance that are \
assessed in the "Evaluating RAG with LlamaIndex" section of the OpenAI Cookbook?'''
]
    for idx, question in enumerate(questions):
        print("Question:", question)
        response = query_engine.query(question)
        response.print_response_stream()
        # print_ref_docs(response.metadata)
        print("\n=="*40)
    

if __name__ == "__main__":
    import fire
    fire.Fire(main)