from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle
from llama_index.core.postprocessor import LLMRerank, SentenceTransformerRerank
import pandas as pd
from IPython.display import display, HTML

# pd.set_option("display.max_colwidth", 100)

def init_reranker(model_name, reranker_top_n=2):

    if model_name == "llm":
        rerank_postprocessor = LLMRerank(
            choice_batch_size=5,
            top_n=reranker_top_n,
        )
    else:
        rerank_postprocessor = SentenceTransformerRerank(
            model='models/mxbai-rerank-xsmall-v1',
            top_n=reranker_top_n, # number of nodes after re-ranking,
            keep_retrieval_score=True
        )
        
    return rerank_postprocessor
        

def get_retrieved_nodes(
    query_str, index, vector_top_k=10, reranker=None
):
    query_bundle = QueryBundle(query_str)
    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=vector_top_k,
    )
    retrieved_nodes = retriever.retrieve(query_bundle)

    if reranker is not None:
        retrieved_nodes = reranker.postprocess_nodes(
            retrieved_nodes, query_bundle
        )

    return retrieved_nodes

def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def visualize_retrieved_nodes(nodes) -> None:
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Metadata": node.metadata, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_print(pd.DataFrame(result_dicts))