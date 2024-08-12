import os
from tqdm import tqdm
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core import (
    Settings,
    VectorStoreIndex, 
    SummaryIndex,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import model_utils, prompt_utils, data_utils
import fire

def load_llm_embed_models(
    llm_name, embed_name, device_map="cuda:1",
    context_window=4096, max_new_tokens=512, temperature=0.2, top_p=0.9
):
    print("Loading embedding model:", embed_name)
    embed_model = HuggingFaceEmbedding(model_name=embed_name, device=device_map)

    print("Loading LLM:", llm_name)
    llm, tokenizer = model_utils.load_quantized_model(
        model_name_or_path=llm_name,
        device=device_map
    )

    prompt_template = "{query_str}"
    if "Meta-Llama-3.1-8B-Instruct" in llm_name:
        prompt_template = prompt_utils.get_llama31_ins_prompt_template()
    elif "Meta-Llama-3-8B-Instruct" in llm_name:
        prompt_template = prompt_utils.get_llama3_ins_prompt_template()

    # print(prompt_template)
    
    llm_hf = HuggingFaceLLM(
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        query_wrapper_prompt=PromptTemplate(prompt_template),
        generate_kwargs={
            "temperature": temperature,
            "do_sample": True,
            "top_p": top_p
        },
        device_map=device_map,
        model_name=llm_name,
        model=llm,
        tokenizer=tokenizer
    )

    print("Loaded LLM and embedding models")
    return llm_hf, embed_model


def get_tools_from_nodes(nodes, doc_metadata, vector_index_root="./database/blogs_md_vector_index"):
    
    blog_title = doc_metadata["title"]
    name_id = doc_metadata["filename"].split(".")[0]
    
    vector_index_dir = f"{vector_index_root}/{name_id}"
    
    # load/build vector index
    if not os.path.exists(vector_index_dir):
        # build vector index
        print(f"Indexing nodes from {name_id} doc")
        vector_index = VectorStoreIndex(nodes)
        vector_index.storage_context.persist(
            persist_dir=vector_index_dir
        )
    else:
        print(f"Found existing index from {name_id}")
        vector_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=vector_index_dir),
        )

    # build summary index
    summary_index = SummaryIndex(nodes)
    
    # define query engines
    vector_query_engine = vector_index.as_query_engine(similarity_top_k=3, use_async=True)
    summary_query_engine = vector_index.as_query_engine(response_mode="tree_summarize", use_async=True)

    # define tools
    # vector tool
    tool_id = name_id.replace("-", "_")
    vector_tool = QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name=f"vector_tool_{tool_id}",
            description=(
                "Useful for questions related to specific aspects of"
                f" {blog_title} (e.g. content, features, techniques,"
            )
        )
    )
    
    # summary tool
    summary_tool = QueryEngineTool(
        query_engine=summary_query_engine,
        metadata=ToolMetadata(
            name=f"summary_tool_{tool_id}",
            description=(
                "Use ONLY IF you want to get a holistic summary"
                f" of EVERYTHING about {blog_title}."
                f"Do NOT use if you have specific questions over {blog_title}."
            )
        )
    )

    return vector_tool, summary_tool


def test_load_models():
    llm_hf, embed_model = load_llm_embed_models(
        llm_name="models/Meta-Llama-3.1-8B-Instruct",
        embed_name="models/bge-base-en-v1.5"
    )

def test_load_tools():

    Settings.llm, Settings.embed_model = load_llm_embed_models(
        llm_name="models/Meta-Llama-3.1-8B-Instruct",
        embed_name="models/bge-base-en-v1.5"
    )
    
    documents = data_utils.load_md_documents(
        docs_dir="data/llama-blogs-md", docs_metadata="data/llama_blogs_metadata.json"
    )
    test_doc = documents[0]
    test_nodes = data_utils.parse_md_doc([test_doc])

    print("Initializing document tools")
    vector_tool, summary_tool = get_tools_from_nodes(test_nodes, doc_metadata=test_doc.metadata)

    # import pdb;pdb.set_trace()
    print("Loaded tools:", vector_tool.metadata.name, summary_tool.metadata.name, sep="\n")
    
    
if __name__ == "__main__":
    fire.Fire()