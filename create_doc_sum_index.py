# logging setup
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# llama index ascyncio config
import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex, StorageContext, Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import chromadb
import model_utils
import prompt_utils
import fire


def main(local_db_dir, collection_name):
    # load embeddings
    embed_model = HuggingFaceEmbedding(model_name="models/bge-small-en-v1.5", device="cuda")
    
    # load llm
    model_name = "models/zephyr-7b-beta"
    model, tokenizer = model_utils.load_quantized_model(
        model_name_or_path=model_name,
        device="cuda"
    )
    # Set `pad_token_id` to `eos_token_id`
    model.generation_config.pad_token_id = model.generation_config.eos_token_id
    
    # config llm and embed_model to llamaindex
    llm_hf = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=512,
        query_wrapper_prompt=PromptTemplate("<|system|>\n</s>\n<|user|>\n{query_str}</s>\n<|assistant|>\n"),
        generate_kwargs={
            "temperature": 0.7,
            "top_k": 50, 
            "top_p": 0.95,
            "do_sample": True
        },
        device_map="cuda",
        model_name=model_name,
        model=model,
        messages_to_prompt=prompt_utils.zephyr_messages_to_prompt,
        tokenizer=tokenizer
    )
    
    # load documents
    documents = SimpleDirectoryReader(
        input_dir="./data",
        filename_as_id=True,
    ).load_data()
    
    print(f"Loaded {len(documents)} documents")
    
    # Creates a persistent instance of Chroma that saves to disk
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    
    # Get or create a collection with the given name and metadata.
    vector_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(
        chroma_collection=vector_collection,
        persist_dir=local_db_dir
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    
    # Init document splitter
    splitter = SentenceSplitter(
        tokenizer=tokenizer,
        chunk_size=1024
    )
    
    # Create document summary index
    response_synthesizer = get_response_synthesizer(
        llm=llm_hf,
        response_mode="tree_summarize", 
        use_async=True,
        verbose=True
    )
    
    doc_summary_index = DocumentSummaryIndex.from_documents(
        documents=documents,
        llm=llm_hf,
        embed_model=embed_model,
        transformations=[splitter],
        response_synthesizer=response_synthesizer,
        show_progress=True,
        storage_context=storage_context
    )
    
    print("Created and saved doc_summary_index to chomadb")
    
    # backup document summary to local storage
    backup_dir = "./backup/"+local_db_dir.split("/")[-1]
    os.makedirs(backup_dir, exist_ok=True)
    doc_summary_index.storage_context.persist(backup_dir)
    
    print("Saved documents summary to backup dir:", backup_dir)

if __name__ == "__main__":
    fire.Fire(main)