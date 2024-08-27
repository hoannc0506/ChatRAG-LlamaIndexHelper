import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from IPython.display import display, Markdown
import tiktoken
import random
import data_utils
import fire


def main(
    save_path,
    test_size=10,
):
    # generator with openai models
    Settings.llm = OpenAI(model="gpt-4o-mini")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

    # set up token counter
    token_counter = TokenCountingHandler(
        tokenizer=tiktoken.encoding_for_model("gpt-4o-mini").encode,
        verbose=True
    )
    Settings.callback_manager = CallbackManager([token_counter])
    
    # load documents
    documents = data_utils.load_md_documents(
        docs_dir='./data/llama-blogs-md',
        docs_metadata='./data/llama_blogs_metadata.json'
    )
    
    # use random doc only to reduce embedding cost
    test_docs = random.sample(documents, test_size)
    
    generator = TestsetGenerator.from_llama_index(
        generator_llm=Settings.llm,
        critic_llm=Settings.llm,
        embeddings=Settings.embed_model,
    )
    
    testset = generator.generate_with_llamaindex_docs(
        test_docs,
        test_size=test_size,
        distributions={simple: 0.4, reasoning: 0.3, multi_context: 0.3},
        with_debugging_logs=True
    )
    
    
    df_test = testset.to_pandas()
    df.to_csv(save_path)
    
    print(
        "Embedding Tokens: ",
        token_counter.total_embedding_token_count,
        "\n",
        "LLM Prompt Tokens: ",
        token_counter.prompt_llm_token_count,
        "\n",
        "LLM Completion Tokens: ",
        token_counter.completion_llm_token_count,
        "\n",
        "Total LLM Token Count: ",
        token_counter.total_llm_token_count,
        "\n",
    )

if __name__ == "__main__":
    fire.Fire(main)






